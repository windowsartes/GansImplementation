# GANs implementation

- [GANs implementation](#gans-implementation)
  - [Overview](#overview)
  - [Stack](#stack)
  - [Testing](#testing)
  - [License](#license)
  - [Credits](#credits)

## Overview

In this repository you can find implementation of some GAN written from scratch on pytorch.
For now, there you can find such architectures, as:
* [Deep Convolutional (DC) GAN](./src/DCGAN/);
* [Conditional Variation of DC GAN](./src/Conditional_GAN/) that works with finit number of classes;
* [Wasserstein GAN](./src/WGAN/);
* [Wasserstein GAN with gradient penalty](./src/WGAN_GP/);
* [Pix2Pix model](./src/Pix2Pix/);


Later I'll also add:
* CycleGAN;
* StyleGAN;
* ProGAN;
* SuperResolution (SR) GAN;
* ESRGAN;

Add a banch of metrics, such as:
* Frechet inception distance;
* Leave-one-out 1NN score;

## Stack

My stack here is:
* Models from scratch via **PyTorch**;
* **Mypy** as a type checker;
* Testing with **pytest**;
* **Ruff** as a linter;

## Testing

All the code was tested on Ubuntu 20.04 and Windownd 11.

## License

Here I use [MIT License](./LICENSE) so feel free to use this code for any your purpose.

## Credits

Many thanks to Aladdin Persson for his awesome [GAN playlist](https://www.youtube.com/playlist?list=PLhhyoLH6IjfwIp8bZnzX8QR30TRcHO8Va) that I find very helpful.