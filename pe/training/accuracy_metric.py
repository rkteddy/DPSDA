import torch
import numpy as np

from pe.callback.callback import Callback
from pe.constant.data import LABEL_ID_COLUMN_NAME
from pe.metric_item import MetricItem


class AccuracyMetric(Callback):
    """Callback that evaluates model accuracy on test data after each PE iteration."""
    
    def __init__(self, model, test_data, device="cuda", batch_size=32):
        """Constructor.
        
        :param model: The model to evaluate
        :type model: torch.nn.Module
        :param test_data: The test data to evaluate on
        :type test_data: :py:class:`pe.data.data.Data`
        :param device: The device to run evaluation on, defaults to "cuda"
        :type device: str, optional
        :param batch_size: Batch size for evaluation, defaults to 32
        :type batch_size: int, optional
        """
        self.model = model
        self.test_data = test_data
        self.device = device
        self.batch_size = batch_size
    
    def __call__(self, syn_data):
        """Evaluate model accuracy on test data.
        
        :param syn_data: The synthetic data (not used for evaluation, but required by callback interface)
        :type syn_data: :py:class:`pe.data.data.Data`
        :return: Test accuracy metric
        :rtype: list[:py:class:`pe.metric_item.MetricItem`]
        """
        self.model.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for i in range(0, len(self.test_data.data_frame), self.batch_size):
                batch_end = min(i + self.batch_size, len(self.test_data.data_frame))
                images = torch.tensor(np.stack(
                    self.test_data.data_frame.iloc[i:batch_end]["PE.IMAGE"].values
                )).to(self.device)
                images = images.permute(0, 3, 1, 2)
                images = images.float() / 255.0
                
                labels = torch.tensor(
                    self.test_data.data_frame.iloc[i:batch_end][LABEL_ID_COLUMN_NAME].values
                ).to(self.device)
                
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                correct += predicted.eq(labels).sum().item()
                total += labels.size(0)
        
        accuracy = correct / total
        return [MetricItem("test_accuracy", accuracy)] 