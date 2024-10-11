# DDPM Implementation

This repository contains an implementation of Denoising Diffusion Probabilistic Models (DDPM) using PyTorch. The code includes modules for position embedding, attention blocks, down-sampling, up-sampling, and a U-Net architecture.

Note: This model has a very large number of parameters. It is recommended to run it on a server.

# Usage

# Training
To train the DDPM model, run the following command:

python DDPM.py

# Detailed Description of Functions

# PositionEmbedding

This class generates position embeddings using sine and cosine functions.

# AttentionBlock

This class implements a down-sampling module.

# down_sample

This class implements a down-sampling module.

# up_sample

This class implements an up-sampling module.

# ResBlk

This class implements a residual block with optional time and class embeddings.

# Unet

This class implements the U-Net architecture with down-sampling, up-sampling, and time embeddings.

# time_extract

This function extracts values from a tensor v based on indices t.

# GaussianDiffusionTrainer

This class implements the training process for the Gaussian diffusion model.

# GaussianDiffusionSampler

This class implements the sampling process for the Gaussian diffusion model.


# Acknowledgements

This implementation is inspired by various research papers and open-source projects on diffusion models.
