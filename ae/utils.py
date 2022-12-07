import random
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def set_deterministic():
    """ Set CUDA deterministic """
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.set_deterministic(True)
   

def set_all_seeds(seed):
    " Initialize all seeds "
    os.environ["PL_GLOBAL_SEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    
def plot_loss(minibatch_loss, num_epochs, averaging_iterations=100, custom_label=''):
    """ Plot minibatch loss over iterations and epochs"""
    iter_per_epoch = len(minibatch_loss) // num_epochs
    plt.figure()
    ax1 = plt.subplot(1, 1, 1)
    ax1.plot(range(len(minibatch_loss)), (minibatch_loss), label=f'Minibatch Loss{custom_label}')
    ax1.set_xlabel('Iterations')
    ax1.set_ylabel('Loss')

    if len(minibatch_loss) < 1000:
        num_losses = len(minibatch_loss) // 2
    else:
        num_losses = 1000

    ax1.set_ylim([0, np.max(minibatch_loss[num_losses:])*1.5])
    ax1.plot(np.convolve(minibatch_loss, np.ones(averaging_iterations,)/averaging_iterations, mode='valid'), label=f'Running Average{custom_label}')
    ax1.legend()

    ax2 = ax1.twiny()
    newlabel = list(range(num_epochs+1))
    newpos = [e*iter_per_epoch for e in newlabel]
    ax2.set_xticks(newpos[::10])
    ax2.set_xticklabels(newlabel[::10])
    ax2.xaxis.set_ticks_position('bottom')
    ax2.xaxis.set_label_position('bottom')
    ax2.spines['bottom'].set_position(('outward', 45))
    ax2.set_xlabel('Epochs')
    ax2.set_xlim(ax1.get_xlim())

    plt.tight_layout()
    

def plot_generated_images(loader, model, device, unnormalizer=None, figsize=(20, 2.5), n_images=15, modeltype='autoencoder'):
    """ Plots generated images """
    
    fig, axes = plt.subplots(nrows=2, ncols=n_images,  sharex=True, sharey=True, figsize=figsize)
    
    for batch_idx, (features, _) in enumerate(loader):
        
        features = features.to(device)

        color_channels = features.shape[1]
        image_height = features.shape[2]
        image_width = features.shape[3]
        
        with torch.no_grad():
            if modeltype == 'autoencoder':
                decoded_images = model(features)[:n_images]
            elif modeltype == 'VAE':
                _, _, _, decoded_images = model(features)[:n_images]
            else:
                raise ValueError('`modeltype` not supported')

        orig_images = features[:n_images]
        break

    for i in range(n_images):
        for ax, img in zip(axes, [orig_images, decoded_images]):
            curr_img = img[i].detach().to(torch.device('cpu'))        
            if unnormalizer is not None:
                curr_img = unnormalizer(curr_img)

            if color_channels > 1:
                curr_img = np.transpose(curr_img, (1, 2, 0))
                ax[i].imshow(curr_img)
            else:
                ax[i].imshow(curr_img.view((image_height, image_width)), cmap='binary')
                
                
def plot_latent_space_with_labels(num_classes, loader, model, device):
    """ Plots latend space """
    
    d = {i:[] for i in range(num_classes)}
    
    model.eval()
    with torch.no_grad():
        for i, (features, targets) in enumerate(loader):

            features = features.to(device)
            targets = targets.to(device)
            
            embedding = model.encoder(features)

            for i in range(num_classes):
                if i in targets:
                    mask = targets == i
                    d[i].append(embedding[mask].to('cpu').numpy())

    colors = list(mcolors.TABLEAU_COLORS.items())
    for i in range(num_classes):
        d[i] = np.concatenate(d[i])
        plt.scatter(
            d[i][:, 0], d[i][:, 1],
            color=colors[i][1],
            label=f'{i}',
            alpha=0.5)

    plt.legend()