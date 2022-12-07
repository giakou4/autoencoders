# Autoencoders (AE)

## What are Autoencoders

Autoencoder (AE) is a type of neural network where the output layer has the same dimensionality as the input layer. An AE replicates the data from the input to the output in an **unsupervised** manner.

The autoencoders reconstruct each dimension of the input by passing it through the network. The middle layers of the neural network have a fewer number of units as compared to that of input or output layers. Therefore, the middle layers hold the **reduced representation** of the input. The output is reconstructed from this reduced representation of the input.

## Architecture of Autoencoders

An autoencoder consists of three components:

1. **Encoder**: An encoder is a neural network that compresses the input into a latent space representation.

2. **Bottleneck/Latent Space/Code/Embedded Space/Hidden Units**: This part of the network contains the reduced representation of the input that is fed into the decoder.

3. **Decoder**: Decoder is also a neural network like the encoder and has a similar structure to the encoder. This network is responsible for reconstructing the input back to the original dimensions from the code.

4. **Loss**: A reconstruction loss of the original input ```x```: ```L(x, x')``` which measures the differences between our original input and the consequent reconstruction.

First, the input goes through the encoder where it is compressed and stored in the latent space, then the decoder decompresses the original input from the code. The main objective of the autoencoder is to get an output identical to the input.

Note that the decoder architecture is the mirror image of the encoder. This is not a requirement but it’s typically the case. **The only requirement is the dimensionality of the input and output must be the same**.


## Types of autoencoders

There are many types of AE and some of them are mentioned below with a brief description:

1. **Fully-Connected (Multi-layer Perceptron) Autoencoder (MLP-AE)**: Autoencoders learn to encode the input in a set of simple signals and then reconstruct the input from them. If we don't use
non-linear activation functions and minimize the MSE, this is very similar to PCA. However, the latent dimensions will not necessarily be orthogonal and will have ~ same variance. The loss used is either  ``` L(x,x') = MSE(x, x') = ||x-x'||2 ^ 2 ``` or ``` L(x,x') = L2(x, x') = ||x-x'||2``` where ```x' = Dec(Enc(x))```

    Applications:
    1. Use embedding as input to classic machine learning methods (SVM, KNN, Random Forest, ...)
    2. Similar to transfer learning, train autoencoder on large image dataset, then fine tune encoder part on your own, smaller dataset
    3. Latent space can also be used for visualization, but better methods exist (e.g., tSNE, EDA, clustering)

2. **Convolutional Autoencoder (CAE)**: Convolutional Autoencoders learn to encode the input in a set of simple signals and then reconstruct the input from them. In this type of AE, encoder layers are known as convolution layers (usually conist of ```Conv2d```, ```LReLU```, ```BatchNorm2d```, ```Dropout2d``` and decoder layers are also called deconvolution layers (usually conist of ```ConvTranspose2d```, ```LReLU```, ```BatchNorm2d```, ```Dropout2d``` or ```UpsamplingNearest2d```, ```Conv2d```, ```BatchNorm2d```, ```LReLU```, ```Dropout2d```. The deconvolution side is also known as upsampling or transpose convolution. The loss used is either ``` L(x,x') = MSE(x, x') = ||x-x'||2 ^ 2 ```  or ``` L(x,x') = BCE(x, x')``` (not symmetric though)

3. **Variational Autoencoders (VAE)**: This type of autoencoder can generate new images just like Generative Adversarial Networks (GANs). VAE models tend to make strong assumptions related to the distribution of latent variables. They use a variational approach for latent representation learning, which results in an additional loss component and a specific estimator for the training algorithm called the Stochastic Gradient Variational Bayes estimator. The probability distribution of the latent vector of a variational autoencoder typically matches the training data much closer than a standard autoencoder. As VAEs are much more flexible and customisable in their generation behaviour than GANs, they are suitable for art generation of any kind. The loss used is ```L(x,x') = L2(x,x') + KL(x)``` or ```MSE``` instead of ```L2```

4. **Denoising autoencoders**: Denoising autoencoders add some noise to the input image and learn to remove it. These AE take a partially corrupted input while training to recover the original undistorted input. The model learns a vector field for mapping the input data towards a lower-dimensional manifold which describes the natural data to cancel out the added noise. By this means, the encoder will extract the most important features and learn a more robust representation of the data. Note that it can be done by adding ```Dropout2d``` after the input. The loss used is either ``` L(x,x') = MSE(x, x')``` or ``` L(x,x') = L2norm(x, x')```

5. **Sparse Autoencoder**: More than half of the representations are zero. It can be done by adding L1 pentalty to the loss to learn sparse feature representations. The loss used is ``` L(x,x') = MSE(x, x') + L1penalty(x) = MSE(x, x') + Σ | Enc(x) |```

6. **Deep autoencoders**: A deep autoencoder is composed of two symmetrical deep-belief networks having four to five shallow layers. One of the networks represents the encoding half of the net and the second network makes up the decoding half. They have more layers than a simple autoencoder and thus are able to learn more complex features. The layers are restricted Boltzmann machines, the building blocks of deep-belief networks.

7. **Contractive autoencoders**: One would expect that for very similar inputs, the learned encoding would also be very similar. We can explicitly train our model in order for this to be the case by requiring that the derivative of the hidden layer activations are small with respect to the input. In other words, for small changes to the input, we should still maintain a very similar encoded state. We can accomplish this by constructing a loss term which penalizes large derivatives of our hidden layer activations with respect to the input training examples. In fancier mathematical terms, we can craft our regularization loss term as the squared Frobenius norm ```||A||``` of the Jacobian matrix ```J``` for the hidden layer activations with respect to the input observations.

## Application of Autoencoders

1. **Data Compression**: Although AE are designed for data compression yet they are hardly used for this purpose in practical situations. The reasons are:
    * Lossy compression: The output of the AE is not exactly the same as the input, it is a close but degraded representation.
    * Data-specific: AE are only able to meaningfully compress data similar to what they have been trained on, since they learn features specific for the given training data.

2. **Image Denoising**: AE are very good at denoising images: when an image gets corrupted or there is a bit of noise in it.

3. **Dimensionality Reduction**: The autoencoders convert the input into a reduced representation which is stored in the middle layer called code. This is where the information from the input has been compressed and by extracting this layer from the model, each node can now be treated as a variable. Thus we can conclude that by trashing out the decoder part, an autoencoder can be used for dimensionality reduction with the output being the code layer.

4. **Feature Extraction**: Encoding part of Autoencoders helps to learn important hidden features present in the input data, in the process to reduce the reconstruction error. 

5. **Image Generation**: VAE discussed above is a Generative Model, used to generate images that have not been seen by the model yet. 

6. **Image colourisation**: One of the applications of autoencoders is to convert a black and white picture into a coloured image. 

7. **Image Inpainting**: Image Inpainting is a task of reconstructing missing regions in an image.

## More Information

Course website: http://pages.stat.wisc.edu/~sraschka/teaching/stat453-ss2021/

GitHub repository: https://github.com/rasbt/stat453-deep-learning-ss21/tree/main/L16
