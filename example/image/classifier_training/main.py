import os
import torch
import numpy as np
import torchvision.models as models
from torch.optim import SGD

from pe.data import Cifar10
from pe.runner import PE
from pe.histogram import GradientNeighbors
from pe.training import TrainingCallback, AccuracyMetric
from pe.population import PEPopulation
from pe.api.image import ImprovedDiffusion270M
from pe.callback import SaveCheckpoints
from pe.logger import CSVPrint
from pe.logger import LogPrint
from pe.logging import setup_logging

    
if __name__ == "__main__":
    exp_folder = "results/classifier_training/cifar10_resnet"
    os.makedirs(exp_folder, exist_ok=True)
    setup_logging(log_file=os.path.join(exp_folder, "log.txt"))

    model = models.resnet50(pretrained=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 10)
    model = model.to("cuda")
    optimizer = SGD(model.parameters(), lr=1e-1, momentum=0.9)
    loss_fn = torch.nn.CrossEntropyLoss()

    data = Cifar10()
    test_data = Cifar10(split="test")

    api = ImprovedDiffusion270M(
        variation_degrees=list(range(0, 42, 2)),
        timestep_respacing="100",
    )

    histogram = GradientNeighbors(
        model=model,
        num_nearest_neighbors=1,
        batch_size=32,
        loss_fn=loss_fn,
        device="cuda"
    )

    population = PEPopulation(api=api, histogram_threshold=20)

    save_checkpoints = SaveCheckpoints(os.path.join(exp_folder, "checkpoint"))
    compute_accuracy = AccuracyMetric(model, test_data)
    training_callback = TrainingCallback(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs_per_iteration=5,
        batch_size=32
    )
    
    csv_print = CSVPrint(output_folder=exp_folder)
    log_print = LogPrint()

    pe_runner = PE(
        priv_data=data,
        population=population,
        histogram=histogram,
        callbacks=[save_checkpoints, compute_accuracy, training_callback],
        loggers=[csv_print, log_print]
    )
    pe_runner.run(
        num_samples_schedule=[50] * 21,
        delta=1e-5,
        noise_multiplier=10 * np.sqrt(2),
        checkpoint_path=os.path.join(exp_folder, "checkpoint"),
    )

