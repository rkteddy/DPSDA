import torch
import numpy as np
import os
from collections import Counter
import copy

from pe.histogram import Histogram
from pe.constant.data import CLEAN_HISTOGRAM_COLUMN_NAME
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.constant.data import HISTOGRAM_NEAREST_NEIGHBORS_VOTING_IDS_COLUMN_NAME
from pe.logging import execution_logger


class GradientNeighbors(Histogram):
    """Compute histogram based on gradient similarity between private and synthetic samples. Each private sample will 
    vote for their closest `num_nearest_neighbors` synthetic samples based on gradient similarity to construct the 
    histogram. The l2 norm of the votes from each private sample is normalized to 1.
    """

    def __init__(
        self,
        model,
        num_nearest_neighbors=1,
        batch_size=32,
        loss_fn=None,
        device="cuda",
        gradient_similarity_metric="cosine",
        voting_details_log_folder=None,
    ):
        """Constructor.

        :param model: The downstream model to compute gradients with respect to
        :type model: torch.nn.Module
        :param num_nearest_neighbors: The number of nearest neighbors to consider for each private sample, defaults to 1
        :type num_nearest_neighbors: int, optional
        :param batch_size: The batch size for processing samples, defaults to 32
        :type batch_size: int, optional
        :param loss_fn: The loss function to use for computing gradients. If None, CrossEntropyLoss is used. Defaults to None
        :type loss_fn: torch.nn.Module, optional
        :param device: The device to run computations on, defaults to "cuda"
        :type device: str, optional
        :param gradient_similarity_metric: The similarity metric to use for comparing gradients. Should be one of
            "cosine" (cosine similarity) or "l2" (negative l2 distance). Defaults to "cosine"
        :type gradient_similarity_metric: str, optional
        :param voting_details_log_folder: The folder to save the logs of the voting details. If it is None, the logs
            are not saved. Defaults to None
        :type voting_details_log_folder: str, optional
        :raises ValueError: If the gradient_similarity_metric is unknown
        :raises ValueError: If the device is not available
        """
        super().__init__()
        self._model = model
        self._num_nearest_neighbors = num_nearest_neighbors
        self._batch_size = batch_size
        self._loss_fn = loss_fn or torch.nn.CrossEntropyLoss()
        self._device = device
        self._gradient_similarity_metric = gradient_similarity_metric
        self._voting_details_log_folder = voting_details_log_folder

        if self._gradient_similarity_metric not in ["cosine", "l2"]:
            raise ValueError(f"Unknown gradient similarity metric: {self._gradient_similarity_metric}")

        if self._device == "cuda" and not torch.cuda.is_available():
            execution_logger.warning("CUDA is not available, falling back to CPU")
            self._device = "cpu"

        # Move model to device
        self._model = self._model.to(self._device)

    def _log_voting_details(self, priv_data, syn_data, ids):
        """Log the voting details.

        :param priv_data: The private data
        :type priv_data: :py:class:`pe.data.data.Data`
        :param syn_data: The synthetic data
        :type syn_data: :py:class:`pe.data.data.Data`
        :param ids: The IDs of the nearest neighbors for each private sample
        :type ids: np.ndarray
        """
        if self._voting_details_log_folder is None:
            return
        labels = set(list(priv_data.data_frame[LABEL_ID_COLUMN_NAME].values))
        assert len(labels) == 1
        label = list(labels)[0]
        iteration = syn_data.metadata["iteration"]
        log_folder = os.path.join(self._voting_details_log_folder, f"{iteration}", f"label-id{label}")
        priv_data = copy.deepcopy(priv_data)
        priv_data.data_frame[HISTOGRAM_NEAREST_NEIGHBORS_VOTING_IDS_COLUMN_NAME] = list(ids)
        priv_data.save_checkpoint(log_folder)

    def _compute_gradient(self, images, labels):
        """Compute gradients for model parameters with respect to each image.

        :param images: Batch of images as torch tensor
        :type images: torch.Tensor
        :param labels: Corresponding labels as torch tensor
        :type labels: torch.Tensor
        :return: Gradients for each sample in the batch
        :rtype: torch.Tensor
        """
        for param in self._model.parameters():
            param.requires_grad = True

        images = images.to(self._device)
        labels = labels.to(self._device)

        if len(images.shape) == 4 and images.shape[-1] in [1, 3]:  # (B, H, W, C)
            images = images.permute(0, 3, 1, 2).float() / 255.0
        else:
            images = images.float()

        sample_grads = []
        for i in range(len(images)):
            self._model.zero_grad()
            output = self._model(images[i:i+1])
            loss = self._loss_fn(output, labels[i:i+1])
            loss.backward()

            grad = torch.cat([p.grad.flatten() for p in self._model.parameters() if p.grad is not None])
            sample_grads.append(grad.detach().cpu())

        return torch.stack(sample_grads)

    def _compute_similarity(self, priv_grads, syn_grads):
        """Compute similarity between private and synthetic gradients.

        :param priv_grads: Private gradients
        :type priv_grads: torch.Tensor
        :param syn_grads: Synthetic gradients
        :type syn_grads: torch.Tensor
        :return: Similarity matrix between private and synthetic gradients
        :rtype: torch.Tensor
        """
        if self._gradient_similarity_metric == "cosine":
            priv_grads_norm = priv_grads / (priv_grads.norm(dim=1, keepdim=True) + 1e-8)
            syn_grads_norm = syn_grads / (syn_grads.norm(dim=1, keepdim=True) + 1e-8)
            similarity_matrix = torch.mm(priv_grads_norm, syn_grads_norm.t())
        elif self._gradient_similarity_metric == "l2":
            similarity_matrix = -torch.cdist(priv_grads, syn_grads)
        else:
            raise ValueError(f"Unknown gradient similarity metric: {self._gradient_similarity_metric}")

        return similarity_matrix

    def _compute_gradients_batch(self, data_frame, data_type="private"):
        """Compute gradients for a batch of data.

        :param data_frame: DataFrame containing images and labels
        :type data_frame: pandas.DataFrame
        :param data_type: Type of data being processed (for logging), defaults to "private"
        :type data_type: str, optional
        :return: Computed gradients for all samples
        :rtype: torch.Tensor
        """
        all_grads = []

        for i in range(0, len(data_frame), self._batch_size):
            execution_logger.debug(f"Computing gradients for {data_type} samples {i}-{min(i+self._batch_size-1, len(data_frame)-1)}")
            
            batch_images = torch.tensor(np.stack(
                data_frame.iloc[i:i+self._batch_size]["PE.IMAGE"].values
            ))
            batch_labels = torch.tensor(
                data_frame.iloc[i:i+self._batch_size][LABEL_ID_COLUMN_NAME].values
            )
            
            batch_grads = self._compute_gradient(batch_images, batch_labels)
            all_grads.append(batch_grads)

        return torch.cat(all_grads, dim=0)

    def compute_histogram(self, priv_data, syn_data):
        """Compute histogram based on gradient similarity between private and synthetic samples.

        :param priv_data: The private data
        :type priv_data: :py:class:`pe.data.data.Data`
        :param syn_data: The synthetic data
        :type syn_data: :py:class:`pe.data.data.Data`
        :return: The private data and the synthetic data, with the computed histogram in the column
            :py:const:`pe.constant.data.CLEAN_HISTOGRAM_COLUMN_NAME`
        :rtype: tuple[:py:class:`pe.data.data.Data`, :py:class:`pe.data.data.Data`]
        """
        execution_logger.info(
            f"Histogram: computing gradient-based nearest neighbors histogram for {len(priv_data.data_frame)} private "
            f"samples and {len(syn_data.data_frame)} synthetic samples using {self._gradient_similarity_metric} similarity"
        )

        self._model.eval()

        execution_logger.info("Computing gradients for private samples")
        priv_grads = self._compute_gradients_batch(priv_data.data_frame, "private")

        execution_logger.info("Computing gradients for synthetic samples")
        syn_grads = self._compute_gradients_batch(syn_data.data_frame, "synthetic")

        execution_logger.info("Computing gradient similarity matrix")
        similarity_matrix = self._compute_similarity(priv_grads, syn_grads)

        execution_logger.info(f"Finding top {self._num_nearest_neighbors} nearest neighbors for each private sample")
        _, top_k_indices = similarity_matrix.topk(self._num_nearest_neighbors, dim=1)

        ids = top_k_indices.numpy()
        self._log_voting_details(priv_data=priv_data, syn_data=syn_data, ids=ids)

        execution_logger.info("Computing histogram from voting results")
        counter = Counter(list(ids.flatten()))
        count = np.zeros(shape=len(syn_data.data_frame), dtype=np.float32)
        count[list(counter.keys())] = list(counter.values())
        count /= np.sqrt(self._num_nearest_neighbors)

        syn_data.data_frame[CLEAN_HISTOGRAM_COLUMN_NAME] = count

        execution_logger.info(
            f"Histogram: finished computing gradient-based nearest neighbors histogram for {len(priv_data.data_frame)} "
            f"private samples and {len(syn_data.data_frame)} synthetic samples"
        )

        return priv_data, syn_data