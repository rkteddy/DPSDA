import torch
import numpy as np
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.callback.callback import Callback
from pe.metric_item import FloatMetricItem

class ClassifierTrainer:
    """Manages the training loop for the classifier."""
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device="cuda",
        num_epochs_per_iteration=5,
        batch_size=32
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.num_epochs_per_iteration = num_epochs_per_iteration
        self.batch_size = batch_size
        
    def train_iteration(self, syn_data):
        """Train the model for one PE iteration using selected synthetic data."""
        self.model.train()
        
        for epoch in range(self.num_epochs_per_iteration):
            total_loss = 0
            num_batches = 0
            
            indices = np.random.permutation(len(syn_data.data_frame))
            
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                batch_images = torch.tensor(np.stack(
                    syn_data.data_frame.iloc[batch_indices]["PE.IMAGE"].values
                )).to(self.device)
                batch_images = batch_images.permute(0, 3, 1, 2).float() / 255.0
                
                batch_labels = torch.tensor(
                    syn_data.data_frame.iloc[batch_indices]["PE.LABEL_ID"].values
                ).to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_images)
                loss = self.loss_fn(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            avg_loss = total_loss / num_batches
            print(f"Epoch {epoch+1}/{self.num_epochs_per_iteration}, Loss: {avg_loss:.4f}")


class TrainingCallback(Callback):
    """Callback that trains a classifier on synthetic data after each PE iteration."""
    
    def __init__(
        self,
        model,
        optimizer,
        loss_fn,
        device="cuda",
        num_epochs_per_iteration=5,
        batch_size=32
    ):
        """Constructor.
        
        :param model: The PyTorch model to train
        :type model: torch.nn.Module
        :param optimizer: The optimizer for training
        :type optimizer: torch.optim.Optimizer
        :param loss_fn: The loss function to use for training
        :type loss_fn: torch.nn.Module
        :param device: The device to run training on, defaults to "cuda"
        :type device: str, optional
        :param num_epochs_per_iteration: Number of epochs to train per PE iteration, defaults to 5
        :type num_epochs_per_iteration: int, optional
        :param batch_size: Batch size for training, defaults to 32
        :type batch_size: int, optional
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        self.num_epochs_per_iteration = num_epochs_per_iteration
        self.batch_size = batch_size
        
    def __call__(self, syn_data):
        """Train the model on synthetic data and return training metrics.
        
        :param syn_data: The synthetic data to train on
        :type syn_data: :py:class:`pe.data.data.Data`
        :return: Training metrics (final epoch loss)
        :rtype: list[:py:class:`pe.metric_item.FloatMetricItem`]
        """
        self.model.train()
        final_loss = 0
        
        for epoch in range(self.num_epochs_per_iteration):
            total_loss = 0
            num_batches = 0
            
            indices = np.random.permutation(len(syn_data.data_frame))
            
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                batch_images = torch.tensor(np.stack(
                    syn_data.data_frame.iloc[batch_indices]["PE.IMAGE"].values
                )).to(self.device)
                batch_images = batch_images.permute(0, 3, 1, 2).float() / 255.0
                
                batch_labels = torch.tensor(
                    syn_data.data_frame.iloc[batch_indices]["PE.LABEL_ID"].values
                ).to(self.device)
                
                self.optimizer.zero_grad()
                outputs = self.model(batch_images)
                loss = self.loss_fn(outputs, batch_labels)
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            avg_loss = total_loss / num_batches
            final_loss = avg_loss
            print(f"Epoch {epoch+1}/{self.num_epochs_per_iteration}, Loss: {avg_loss:.4f}")
        
        # Return the final training loss as a metric
        return [FloatMetricItem(name="training_loss", value=final_loss)]