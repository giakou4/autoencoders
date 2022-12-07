import argparse
import torch
import torch.nn.functional as F
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils import set_deterministic, set_all_seeds
from utils import plot_loss, plot_generated_images, plot_latent_space_with_labels, \
                  plot_images_sampled_from_vae, inspect_latent_space
from model import VariationalAutoencoder1 as VariationalAutoencoder


def parse_opt():
    """ Arguement Parser """
    parser = argparse.ArgumentParser(description='Hyperparameters for training')
    parser.add_argument('--device', type=str, default='cuda', help='torch device')
    parser.add_argument('--num-epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--num-workers', type=int, default=2, help='number of workers')
    parser.add_argument('--learning-rate', type=float, default=0.0005, help='base learning rate')
    parser.add_argument('--batch-size', type=int, default=128, help='batch size')
    parser.add_argument('--random-seed', type=int, default=123, help='batch size')
    parser.add_argument('--num-classes', type=int, default=10, help='batch size')
    config = parser.parse_args()
    return config


def train_one_epoch(loader, model, optimizer, loss_fn, epoch, num_epochs, device, w=1):
    """ One forward pass of Autoencoder """
    
    loop = tqdm(loader, leave=True)
    loop.set_description(f"Epoch {epoch+1}/{num_epochs}")
    losses, losses_rec, losses_k1 = [], [], []
    
    for batch_idx, (features, _) in enumerate(loop):
        features = features.to(device)
        
        encoded, z_mean, z_log_var, decoded = model(features)
        
        kl_div = -0.5 * torch.sum(1 + z_log_var - z_mean**2 - torch.exp(z_log_var), axis=1)
        batch_size = kl_div.size(0)
        kl_div = kl_div.mean()
        
        pixelwise = loss_fn(decoded, features, reduction='none')
        pixelwise = pixelwise.view(batch_size, -1).sum(axis=1)
        pixelwise = pixelwise.mean() 
        
        loss = w * pixelwise + kl_div
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        losses_rec.append(pixelwise.item())
        losses_k1.append(kl_div.item())
        
        loop.set_postfix(loss=loss.item(), loss_rec=pixelwise.item(), loss_k1=kl_div.item())
        
    return losses, losses_rec, losses_k1
   

def main(config):
    """ Training of Autoencoder """
    
    set_deterministic
    set_all_seeds(config.random_seed)
    
    transform = transforms.ToTensor()
    
    #dataset = datasets.CelebA(root='data', split='train', transform=transform, download=True)
    dataset = datasets.MNIST(root='data', train=True, transform=transform, download=True)
    loader = DataLoader(dataset=dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
    
    model = VariationalAutoencoder()
    model.to(config.device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate) 
    loss_fn = F.mse_loss
    
    model.train()
    losses_arr, losses_rec_arr, losses_k1_arr = [], [], []
    for epoch in range(config.num_epochs):
        losses, losses_rec, losses_k1 = train_one_epoch(loader, model, optimizer, loss_fn, epoch, config.num_epochs, config.device)
        losses_arr.extend(losses)
        losses_rec_arr.extend(losses_rec)
        losses_k1_arr.extend(losses_k1)
    
    plot_loss(losses_arr, config.num_epochs)
    plot_loss(losses_rec_arr, config.num_epochs, custom_label=" (reconstruction)")
    plot_loss(losses_k1_arr, config.num_epochs, custom_label=" (KL)")
    plt.show()
    
    plot_generated_images(loader=loader, model=model, device=config.device, modeltype="VAE")
    plt.show()
    
    plot_latent_space_with_labels(num_classes=config.num_classes, loader=loader, encoding_fn=model.encoding_fn, device=config.device)
    plt.legend()
    plt.show()
    
    for i in range(10):
        plot_images_sampled_from_vae(model=model, device=config.device, latent_size=2)
        plt.show()
    
    #inspect_latent_space(model, loader, config.device, s=2)
    #plt.show()
    
if __name__ == "__main__":
    config = parse_opt()
    main(config)