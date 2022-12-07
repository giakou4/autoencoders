import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import set_deterministic, set_all_seeds
from utils import plot_loss, plot_generated_images, plot_latent_space_with_labels
from model import AutoEncoder


def parse_opt():
    """ Arguement Parser """
    parser = argparse.ArgumentParser(description='Hyperparameters for training')
    parser.add_argument('--device', type=str, default='cuda', help='torch device')
    parser.add_argument('--num-epochs', type=int, default=100, help='number of epochs')
    parser.add_argument('--num-workers', type=int, default=2, help='number of workers')
    parser.add_argument('--learning-rate', type=float, default=0.0005, help='base learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--random-seed', type=int, default=123, help='batch size')
    parser.add_argument('--num-classes', type=int, default=10, help='batch size')
    config = parser.parse_args()
    return config


def train_one_epoch(loader, model, optimizer, loss_fn, epoch, num_epochs, device):
    """ One forward pass of Autoencoder """
    
    loop = tqdm(loader, leave=True)
    loop.set_description(f"Epoch {epoch+1}/{num_epochs}")
    losses = []
    
    for batch_idx, (features, _) in enumerate(loop):
        features = features.to(device)
        
        logits = model(features)
        loss = loss_fn(logits, features)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        loop.set_postfix(loss=loss.item())
        
    return losses
   

def main(config):
    """ Training of Autoencoder """
    
    set_deterministic
    set_all_seeds(config.random_seed)
    
    transform = transforms.ToTensor()
    
    dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
    loader = DataLoader(dataset=dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
    
    model = AutoEncoder()
    model.to(config.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate) 
    loss_fn = F.mse_loss
    
    model.train()
    losses_arr = []
    for epoch in range(config.num_epochs):
        losses = train_one_epoch(loader, model, optimizer, loss_fn, epoch, config.num_epochs, config.device)
        losses_arr.extend(losses)
    
    plot_loss(losses_arr, config.num_epochs)
    plt.show()
    
    plot_generated_images(loader=loader, model=model, device=config.device)
    plt.show()
    
    plot_latent_space_with_labels(num_classes=config.num_classes, loader=loader, model=model, device=config.device)
    plt.legend()
    plt.show()
    
    
if __name__ == "__main__":
    config = parse_opt()
    main(config)