import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
from torchvision import transforms
from models.convnet import ConvNet
from models.resnet9 import ResNet9
import argparse
from utils import train_step, test, get_transform

# Custom Dataset to load image data from the DataFrame loaded from a pickle file
class CustomImageDataset(Dataset):
    def __init__(self, data, transform=None):
        """
        Args:
            data (DataFrame): The DataFrame containing image data, assumed to have 'PE.IMAGE' column for images.
            transform (callable, optional): Optional transformations to apply to images.
        """
        self.data = data
        self.transform = transform

    def __len__(self):
        """Returns the total number of images in the dataset."""
        return len(self.data)

    def __getitem__(self, idx):
        """Fetches an image and its corresponding label from the dataset."""
        image = np.array(self.data['PE.IMAGE'][idx])  # Convert the image data to a NumPy array
        image = np.transpose(image, (2, 0, 1))  # Convert the image from (H, W, C) to (C, H, W)

        # Apply transformations if provided (e.g., resize, normalization)
        if self.transform:
            image = self.transform(image)

        # Get the label for the current image
        label = self.data['PE.IMAGE_MODEL_LABEL'][idx]
        return torch.tensor(image, dtype=torch.float32), torch.tensor(label, dtype=torch.long)


# Argument parsing for command line inputs
def parse_args():
    parser = argparse.ArgumentParser(description="Train a model on CIFAR10 or STL10")
    
    # Model and dataset arguments
    parser.add_argument('--model', type=str, choices=['convnet', 'resnet9'], required=True, help='Model architecture to use')
    parser.add_argument('--dataset', type=str, choices=['CIFAR10', 'STL10'], required=True, help='Dataset to train on')
    parser.add_argument('--train_dir', type=str, required=True, help='Directory for PASDA synthetic data')
    parser.add_argument('--private_dir', type=str, required=True, help='Directory for private data')

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train the model')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of workers for dataloader')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay for optimizer')

    # Image size
    parser.add_argument('--image_size', type=int, default=32, help='Image size to resize input')

    # Device argument
    parser.add_argument('--device', type=str, default='cuda:0', help='Device to use for training, e.g., "cuda:0" or "cpu"')

    return parser.parse_args()


# Main script execution
if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_args()

    # Set device (GPU if available, otherwise CPU)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Initialize the model based on the architecture chosen
    if args.model == 'convnet':
        net_width, net_depth, net_act, net_norm, net_pooling = 128, 3, 'relu', 'instancenorm', 'avgpooling'
        model = ConvNet(channel=3, num_classes=10, net_width=net_width, net_depth=net_depth, net_act=net_act, net_norm=net_norm, net_pooling=net_pooling, im_size=(args.image_size, args.image_size)).to(device)
    elif args.model == 'resnet9':
        model = ResNet9(num_class=10).to(device)

    # Load the data from the pickle file
    data_path = '/data/runkai/DPSDA/results/image/cifar10_eps=1_5k/checkpoint/000000019/data_frame.pkl'
    with open(data_path, 'rb') as file:
        data = pickle.load(file)

    # Create the custom dataset
    train_dataset = CustomImageDataset(data, transform=get_transform(image_size=args.image_size, is_train=True))
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    # Load the test dataset using the provided function (assuming it's defined elsewhere)
    test_loader = get_dataloader(args.dataset, args.batch_size, get_transform(args.image_size, is_train=False), root=args.private_dir, is_train=False)

    # Initialize optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Training loop
    for epoch in range(args.epochs):
        # Perform one training step
        train_loss = train_step(model, train_loader, optimizer, scheduler, device)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {train_loss:.4f}")

        # Perform testing after each epoch
        test_accuracy = test(model, test_loader, device)
        print(f"Test accuracy: {test_accuracy:.4f}")
