# Variational Autoencoder

Course website: http://pages.stat.wisc.edu/~sraschka/teaching/stat453-ss2021/

GitHub repository: https://github.com/rasbt/stat453-deep-learning-ss21/tree/main/L17

## Autoencoder

A regular Autoencoder (AE) minimizees squared error loss ```L = MSE(x, x') = ||x-x'||2 ^2 ```. Autoencoders are cool. But What if we want to "generate" images? What if we want to impose some constraints on the latent space? In our AE, we had no constraints on the latent space. 

<p align="center">
  <img src="https://raw.githubusercontent.com/giakou4/autoencoders/vae/main/assets/ae.png">
</p>

## Variational Autoencoder

We're going to **constrain our latent space to be drawn from a Normal distribution**. Instead of the encoder directly outputting the latent space, its going to output a vector of **means** and a vector of **variances** from which we will sample the latent space from. Thus the loss is the sum of the expected negative log likelihood term (wrt to encoder distribution) and the Kullback-LKeibler (KL) divergence term where ```p(z) = N(mean=0, sigma=1)```:

<p align="center">
  <img src="https://raw.githubusercontent.com/giakou4/vae/main/assets/loss.png">
</p>


This gives us the following benefits:

1. You have constraints on the latent space, can specify priors on your latent variables
2. It's now a generative model

From the graphical models perspective, this is known as amortized variational inference. The general architecture of VAE is:

<p align="center">
  <img src="https://raw.githubusercontent.com/giakou4/vae/main/assets/vae.png">
</p>


### Sampling

A d-dimensional probability density for multivariate Gaussian:

<p align="center">
  <img src="https://raw.githubusercontent.com/giakou4/vae/main/assets/pdf_mult_gauss.png">
</p>

How to sample from a VAE: ```z = mi + sigma * e``` where ``` sigma = [sigma1, sigma2]```, ``` mi = [mi1, mi2]``` and ```e1, e2 ~ N(0,1)``` since VAE's assume a diagonal covariance matrix, thus, we only need a mean and a variance vector, no covariance matrix. As a result, we only need ```mi``` and ```sigma``` to draw from the distribution.

The neural network learns (included in training and back propagation) ```mi = [mi1, mi2]``` and ```sigma = [sigma1, sigma2]```.

<p align="center">
  <img src="https://raw.githubusercontent.com/giakou4/vae/main/assets/vae2.png">
</p>

In the end, we calculate the loss between ```x``` and ```x'``` and the loss from the latent space ```mi```, ```sigma```.

In VAE however, we calculate the log variance (```log(sigma^2)```) instead of variance (```sigma^2```) for stable training, thus instead of ```z = mi + sigma * e``` we use ```z = mi + e^(log(sigma^2)/2) * e```

### Loss

Now we are learning the parameters of a distribution. The encoder learns the parameters for ```q(z|x)``` and our decoder learns the parameters for p(x|z). In our case, we're going to assume that ```p(x|z)``` is Normal. ```p(x)``` is the prior distribution of our latent space, which in our case is going to be Normal.

For our loss term, we sum up two separate losses: the reconstruction loss, which is a mean squared error (MSE) that measures how accurately the network reconstructed the images, and a latent loss, which is the KL divergence that measures how closely the latent variables match a unit gaussian. It minimizes ELBO (Evidence lower bound), consisting of KL term and reconstruction loss.

<p align="center">
  <img src="https://raw.githubusercontent.com/giakou4/vae/main/assets/loss.png">
</p>

For the first term (**reconstruction loss**), the best is to choose MSE that ensures good reconstruction; ```L1 = MSE(x, x') = ||x-x'||2 ^ 2```.

The second term (**KL-divergence loss**) is ```L2 = D-KL[ N(mi, sigma) | N(0, 1) ]``` or ```L2 = -(1/2) sum [ (1 + log(sigma^2) - mi^2 - sigma^2 ]```.

The loss is ```L = L1 + L2```.


### Latent Space Arithmetic

We can give a sad person a smile by ``` z_new = z_orig + a * z_diff```.

<p align="center">
  <img src="https://raw.githubusercontent.com/giakou4/vae/main/assets/lsa.png">
</p>

