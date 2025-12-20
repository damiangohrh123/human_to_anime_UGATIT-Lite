"""
UGATIT training script configured for 512×512 training.
Designed for long runs on a high-end GPU (e.g. RTX 5080). Includes:
 - 512×512 training
 - VGG19 perceptual (relu3_3) loss
 - Hinge GAN loss
 - Identity + cycle losses (tunable weights)
 - Gradient accumulation to handle small per-GPU batch sizes
 - Mixed precision (AMP)
 - Linear LR decay schedule (starting at --lr_decay_start)
 - Replay buffer, checkpointing, sample outputs

Run example:
    Customized parameters:
    python train_ugatit.py --human_dir data/processed/trainA --anime_dir data/processed/trainB --img_size 512 --bs 2 --accum_steps 4 --num_workers 8 --epochs 200
    
    Default parameters:
    python train_ugatit.py --human_dir data/processed/trainA --anime_dir data/processed/trainB
"""

import os
import random
import argparse
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms
from torchvision.utils import save_image
from torchvision import models
from torch.amp import autocast, GradScaler
from torch.nn.utils import spectral_norm

# Save a visual comparison of real input images and their generated outputs
# during training (for qualitative monitoring of generator progress).
def save_samples(epoch, G_AB, dataloader, device, out_dir):
    G_AB.eval() # set to eval mode
    os.makedirs(out_dir, exist_ok=True) # Ensure directory exists, create it if not

    # Disable gradient computation for inference
    with torch.no_grad():
        # Get first batch of data, only use first 4 images, then move to GPU
        real_A, _ = next(iter(dataloader))
        real_A = real_A[:4].to(device)

        # Generate fake images
        fake_B, _ = G_AB(real_A) 

        # Rescale from [-1, 1] to [0, 1]
        real_A = (real_A + 1) * 0.5
        fake_B = (fake_B + 1) * 0.5

        # Concatenate real and fake images along the batch dimension
        grid = torch.cat([real_A, fake_B], dim=0)

        # save the concatenated images as a single PNG file
        save_image(
            grid,
            os.path.join(out_dir, f"epoch_{epoch}.png"),
            nrow=4
        )
    # return to train mode
    G_AB.train()

# Make CuDNN fast
torch.backends.cudnn.benchmark = True

# Dataset
class FaceAnimeDataset(Dataset): # Inherits from PyTorch Dataset
    def __init__(self, human_dir, anime_dir, transform=None):
        # List all files in directory, filter only the files (ignoring subdirectories), sort filenames alphabetically
        self.human_files = sorted([f for f in os.listdir(human_dir) if os.path.isfile(os.path.join(human_dir,f))])
        self.anime_files = sorted([f for f in os.listdir(anime_dir) if os.path.isfile(os.path.join(anime_dir,f))])

        # Store directory paths and transform
        self.human_dir = human_dir
        self.anime_dir = anime_dir
        self.transform = transform

    # Return the smaller length between human and anime datasets
    # Important for UGATIT which used unpaired training
    def __len__(self):
        return min(len(self.human_files), len(self.anime_files))

    # Fetching one sample (1 human face, and 1 anime face)
    def __getitem__(self, idx):
        # Construct full file paths
        human_path = os.path.join(self.human_dir, self.human_files[idx])
        anime_path = os.path.join(self.anime_dir, self.anime_files[idx])

        # Open images and convert to RGB
        human = Image.open(human_path).convert("RGB")
        anime = Image.open(anime_path).convert("RGB")

        # Apply transforms if provided
        if self.transform:
            human = self.transform(human)
            anime = self.transform(anime)
        return human, anime

# Building blocks
class ConvBlock(nn.Module):
    """
    A modular convolutional block.
    Consists of: Convolution -> Normalization -> Activation
    Flexible: supports InstanceNorm, BatchNorm, or no norm, and ReLU, LeakyReLU, or linear activation.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, stride=1, padding=1, norm='in', activation='relu'):
        super().__init__()

        # 1. Convolution layer
        self.conv = nn.Conv2d(
            in_ch, 
            out_ch, 
            kernel_size, 
            stride, 
            padding
        )

        # 2. Normalization layer
        if norm == 'in':      # Instance Normalization
            self.norm = nn.InstanceNorm2d(out_ch, affine=True)
        elif norm == 'bn':    # Batch Normalization
            self.norm = nn.BatchNorm2d(out_ch)
        else:                 # No normalization
            self.norm = nn.Identity()

        # 3. Activation layer
        if activation == 'relu':  
            self.act = nn.ReLU(inplace=True)
        elif activation == 'lrelu':  # LeakyReLU
            self.act = nn.LeakyReLU(0.2, inplace=True)
        else:                        # Linear / no activation
            self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.conv(x)))

class ResBlock(nn.Module):
    """
    Residual Block (used in UGATIT generator).
    Consists of:
        - Two convolutional layers (ConvBlock)
        - Skip connection (adds input to output)
    Purpose:
        - Helps gradients flow through deep networks
        - Allows learning residuals (small changes) instead of full mappings
    """
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(channels, channels, norm='in', activation='relu'),
            ConvBlock(channels, channels, norm='in', activation='none')
        )

    def forward(self, x):
        return x + self.block(x)

class ILN(nn.Module):
    """
    ILN: Instance-Layer Normalization
    Combines Instance Normalization (IN) and Layer Normalization (LN) dynamically.
    Used in UGATIT to stabilize style transfer while preserving content features.
    """
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        # Learnable blending parameter between IN and LN
        # rho close to 1 → mostly InstanceNorm, close to 0 → mostly LayerNorm
        self.rho = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # Learnable affine parameters (scale and shift)
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps  # small constant (epsilon) to avoid division by zero

    def forward(self, x):
        """
        x: input feature map (B x C x H x W)
        """
        # Compute Instance Norm statistics
        in_mean = x.mean([2,3], keepdim=True)   # mean over H and W
        in_var  = x.var([2,3], keepdim=True)    # variance over H and W

        # Compute Layer Norm statistics
        ln_mean = x.mean([1,2,3], keepdim=True) # mean over C, H, W
        ln_var  = x.var([1,2,3], keepdim=True)  # variance over C, H, W

        # Combine IN and LN using rho
        out = self.rho * ((x - in_mean) / torch.sqrt(in_var + self.eps)) + \
              (1 - self.rho) * ((x - ln_mean) / torch.sqrt(ln_var + self.eps))

        # Apply affine transformation (gamma and beta)
        out = out * self.gamma + self.beta

        return out

class AdaILN(nn.Module):
    """
    Adaptive Instance-Layer Normalization.
    Combines IN and LN like ILN, but applies **adaptive gamma and beta**.
    Used in UGATIT generator for style modulation.
    """
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.eps = eps
        # Learnable blending parameter between IN and LN
        self.rho = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x, gamma, beta):
        # Compute Instance Norm statistics
        in_mean = x.mean([2,3], keepdim=True)
        in_var = x.var([2,3], keepdim=True)

        # Compute Layer Norm statistics
        ln_mean = x.mean([1,2,3], keepdim=True)
        ln_var = x.var([1,2,3], keepdim=True)

        # Blend IN and LN using rho
        out = self.rho * ((x - in_mean) / torch.sqrt(in_var + self.eps)) + \
              (1 - self.rho) * ((x - ln_mean) / torch.sqrt(ln_var + self.eps))
        
        # Apply affine transformation (gamma and beta)
        out = out * gamma + beta

        return out

class CAMBlock(nn.Module):
    """
    Class Activation Map (CAM) block for UGATIT.
    Highlights important regions in feature maps using both
    global average pooling (GAP) and global max pooling (GMP).
    """
    def __init__(self, channels):
        super().__init__()
        # Fully connected layers for GAP and GMP
        self.gap_fc = nn.Linear(channels, 1, bias=False)
        self.gmp_fc = nn.Linear(channels, 1, bias=False)

        # 1x1 convolution to fuse GAP and GMP features
        self.conv1x1 = nn.Conv2d(channels*2, channels, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        b, c, h, w = x.size()

        # Global Average Pooling → vector of size (B, C)
        gap = F.adaptive_avg_pool2d(x, 1).view(b,c)

        # Global Max Pooling → vector of size (B, C)
        gmp = F.adaptive_max_pool2d(x, 1).view(b,c)

        # Compute logits for GAP and GMP
        gap_logit = self.gap_fc(gap)
        gmp_logit = self.gmp_fc(gmp)

        # Get weights of the FC layers
        gap_weight = list(self.gap_fc.parameters())[0]
        gmp_weight = list(self.gmp_fc.parameters())[0]

        # Multiply original feature maps by weights → attention maps
        gap = x * gap_weight.unsqueeze(2).unsqueeze(3)
        gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)

        # Concatenate GAP and GMP attention maps along channel dimension
        cam = torch.cat([gap, gmp], dim=1)

        # Fuse and apply non-linearity
        cam = self.relu(self.conv1x1(cam))

        # Return attention map and sum of logits (used in loss)
        return cam, gap_logit + gmp_logit

class Generator(nn.Module):
    """
    UGATIT Generator.
    Architecture: Downsampling → Residual blocks → CAM → Upsampling
    Uses CAM to focus on important regions for style translation.
    """
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_blocks=6):
        super().__init__()
        # 1. Downsampling: extract features while reducing spatial size
        self.down = nn.Sequential(
            ConvBlock(input_nc, ngf, 7, 1, 3, norm='in', activation='relu'),  # initial conv  
            ConvBlock(ngf, ngf*2, 4, 2, 1, norm='in', activation='relu'),     # downsample x2
            ConvBlock(ngf*2, ngf*4, 4, 2, 1, norm='in', activation='relu')    # downsample x2
        )

        # 2. Residual blocks: deeper features without changing spatial dimensions
        res_blocks = []
        for _ in range(n_blocks):
            res_blocks.append(ResBlock(ngf*4))
        self.res_blocks = nn.Sequential(*res_blocks)

        # 3. CAM: focuses on important regions of the feature maps
        self.cam = CAMBlock(ngf*4)

        # 4. Upsampling: reconstruct image from feature maps
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),                      # upsample x2
            ConvBlock(ngf*4, ngf*2, 3, 1, 1, norm='in', activation='relu'),   # conv after upsample
            nn.Upsample(scale_factor=2, mode='nearest'),                      # upsample x2
            ConvBlock(ngf*2, ngf, 3, 1, 1, norm='in', activation='relu'),     # conv after upsample
            nn.Conv2d(ngf, output_nc, 7, 1, 3),                               # final output conv
            nn.Tanh()                                                         # output in [-1, 1]
        )

    def forward(self, x):
        x = self.down(x)              # Downsample   
        x = self.res_blocks(x)        # Residual processing
        x, cam_logit = self.cam(x)    # CAM attention
        x = self.up(x)                # Upsample to original image size
        return x, cam_logit           # returns generated image and CAM logits

class Discriminator(nn.Module):
    """
    PatchGAN Discriminator used in UGATIT.
    Classifies each patch of the image as real or fake.
    """
    def __init__(self, input_nc=3, ndf=64):
        super().__init__()

        # Sequential model: stack of conv layers
        self.model = nn.Sequential(
            # 1. Conv layer + Spectral Norm + LeakyReLU
            spectral_norm(nn.Conv2d(input_nc, ndf, 4, 2, 1)),  # downsample, extract features
            nn.LeakyReLU(0.2, inplace=True),

            # 2. Conv layer + Spectral Norm + LeakyReLU
            spectral_norm(nn.Conv2d(ndf, ndf*2, 4, 2, 1)),     # deeper features, downsample
            nn.LeakyReLU(0.2, inplace=True),

            # 3. Conv layer + Spectral Norm + LeakyReLU
            spectral_norm(nn.Conv2d(ndf*2, ndf*4, 4, 2, 1)),   # more features, downsample
            nn.LeakyReLU(0.2, inplace=True),

            # 4. Conv layer + Spectral Norm + LeakyReLU
            spectral_norm(nn.Conv2d(ndf*4, ndf*8, 4, 1, 1)),   # keep spatial size, extract high-level features
            nn.LeakyReLU(0.2, inplace=True), 

            # 5. Final conv layer to produce 1-channel PatchGAN output
            nn.Conv2d(ndf*8, 1, 4, 1, 1)
        )

    def forward(self, x):
        return self.model(x)

class ReplayBuffer:
    """
    Replay buffer used in UGATIT to store previously generated images.
    Helps stabilize GAN training by reusing past fake images instead of always using the latest ones.
    """
    def __init__(self, max_size=50):
        self.max_size = max_size   # Maximum number of images to store
        self.data = []             # List to hold stored images

    def push_and_pop(self, data):
        """
        Adds new images to the buffer and returns a batch for discriminator training.
        Some images are replaced randomly to introduce variation.
        """
        to_return = []
        for img in data:
            img = img.unsqueeze(0)                # Add batch dimension

            # If buffer not full, just add the image
            if len(self.data) < self.max_size:
                self.data.append(img)
                to_return.append(img)
            else:
                # 50% chance: replace a random old image and return it
                if random.random() > 0.5:
                    idx = random.randint(0, self.max_size-1)
                    tmp = self.data[idx].clone()
                    self.data[idx] = img
                    to_return.append(tmp)
                # 50% chance: return current image without storing
                else:
                    to_return.append(img)

        # Concatenate all images along the batch dimension and return
        return torch.cat(to_return, 0)

class VGGPerceptual(nn.Module):
    """
    Computes perceptual loss between two images using a pretrained network VGG19.
    Measures high-level feature differences instead of pixel-wise differences.
    """
    def __init__(self, device):
        super().__init__()
        # Load pretrained VGG19 features
        vgg = models.vgg19(pretrained=True).features
        self.slice = nn.Sequential()

        # Use layers up to relu3_3 (index 16) for feature extraction
        for x in range(17):
            self.slice.add_module(str(x), vgg[x])
        
        # Freeze VGG weights
        for p in self.slice.parameters():
            p.requires_grad = False
        self.slice = self.slice.to(device)

    def forward(self, x, y):
        # Convert inputs from [-1,1] to [0,1] (VGG expects [0,1])
        x = (x + 1) * 0.5
        y = (y + 1) * 0.5

        # Extract features
        fx = self.slice(x)
        fy = self.slice(y)

        # L1 loss between feature maps
        return F.l1_loss(fx, fy)

def discriminator_hinge_loss(real_pred, fake_pred):
    """
    Hinge loss for the discriminator in a GAN.
    Encourages real images to have high scores (>1)
    and fake images to have low scores (<-1).
    
    Args:
        real_pred: discriminator output on real images
        fake_pred: discriminator output on fake images
    Returns:
        Scalar loss for the discriminator
    """
    # Penalize real images predicted below 1
    loss_real = torch.mean(F.relu(1.0 - real_pred))

    # Penalize fake images predicted above -1
    loss_fake = torch.mean(F.relu(1.0 + fake_pred))

    # Average the two
    return 0.5 * (loss_real + loss_fake)

def generator_hinge_loss(fake_pred):
    """
    Hinge loss for the generator in a GAN.
    Encourages generator to produce images that
    the discriminator classifies as real (high score).
    
    Args:
        fake_pred: discriminator output on generated images
    Returns:
        Scalar loss for the generator
    """
    return -torch.mean(fake_pred)

# Training
def train_model(args):
    # 1. Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    # 2. Image transforms for augmentation and normalization
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),                                   # Resize to fixed size
        transforms.RandomHorizontalFlip(p=0.5),                                              # Flip randomly
        transforms.ColorJitter(brightness=0.05, contrast=0.05, saturation=0.05, hue=0.02),   # Slight color jitter
        transforms.ToTensor(),                                                               # Convert to tensor
        transforms.Normalize([0.5]*3, [0.5]*3)                                               # Normalize to [-1, 1]
    ])

    # 3. Dataset and DataLoader
    dataset = FaceAnimeDataset(args.human_dir, args.anime_dir, transform)
    print('Dataset size:', len(dataset))
    steps_per_epoch = min(args.max_steps_per_epoch, len(dataset))
    sampler = RandomSampler(dataset, replacement=True, num_samples=steps_per_epoch)
    dataloader = DataLoader(
        dataset, batch_size=args.bs, sampler=sampler,
        num_workers=args.num_workers, pin_memory=True if device.type=='cuda' else False,
        persistent_workers=True if args.num_workers>0 else False
    )
    print('DataLoader ready.')

    # 4. Models initialization
    G_A2B = Generator().to(device)              # Generator: Human to Anime
    G_B2A = Generator().to(device)              # Generator: Anime to Human
    D_A = Discriminator().to(device)            # Discriminator for Human domain
    D_B = Discriminator().to(device)            # Discriminator for Anime domain

    # Optional perceptual loss (VGG)
    perceptual = VGGPerceptual(device) if args.use_perceptual else None
    if perceptual:
        perceptual.eval()
    
    # Cycle-consistency and identity losses
    criterion_cycle = nn.L1Loss()
    criterion_identity = nn.L1Loss()

    # 5. Optimizers
    g_params = list(G_A2B.parameters()) + list(G_B2A.parameters())
    d_params = list(D_A.parameters()) + list(D_B.parameters())
    optimizer_G = optim.Adam(g_params, lr=args.lr_G, betas=(0.5, 0.999))
    optimizer_D = optim.Adam(d_params, lr=args.lr_D, betas=(0.5, 0.999))

    # 6. Learning rate schedulers
    def lambda_rule(epoch):
        # Linear decay after lr_decay_start
        return 1.0 if epoch < args.lr_decay_start else 1.0 - (epoch - args.lr_decay_start) / max(1, args.epochs - args.lr_decay_start)
    scheduler_G = optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lambda_rule)
    scheduler_D = optim.lr_scheduler.LambdaLR(optimizer_D, lr_lambda=lambda_rule)

    # 7. AMP scaler for mixed precision
    scaler = GradScaler(enabled=args.use_amp)

    # 8. Replay buffers for discriminator training stability
    replay_buffer_A = ReplayBuffer()
    replay_buffer_B = ReplayBuffer()

    # 9. Prepare directories and checkpoint
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    latest_checkpoint = os.path.join(args.save_dir, 'latest.pth')
    start_epoch = 1

    # 10. Resume from checkpoint if available
    if args.resume and os.path.exists(latest_checkpoint):
        print('Loading checkpoint...')
        checkpoint = torch.load(latest_checkpoint, map_location=device)
        G_A2B.load_state_dict(checkpoint['G_A2B'])
        G_B2A.load_state_dict(checkpoint['G_B2A'])
        D_A.load_state_dict(checkpoint['D_A'])
        D_B.load_state_dict(checkpoint['D_B'])
        optimizer_G.load_state_dict(checkpoint['opt_G'])
        optimizer_D.load_state_dict(checkpoint['opt_D'])
        start_epoch = checkpoint.get('epoch',1) + 1
        print(f"Resuming from epoch {start_epoch}")

    # 11. Training loop
    g_steps_per_d = 2  # Train generator more often than discriminator
    for epoch in range(start_epoch, args.epochs+1):
        tqdm.write(f"Epoch {epoch}/{args.epochs}")
        loop = tqdm(enumerate(dataloader), total=steps_per_epoch)
        epoch_G_loss, epoch_D_loss = 0.0, 0.0
        steps = 0

        for step, (real_A, real_B) in loop:
            real_A = real_A.to(device)
            real_B = real_B.to(device)

            # Train Generators
            for _ in range(g_steps_per_d):
                if scaler:
                    optimizer_G.zero_grad(set_to_none=True)
                    with autocast(device_type='cuda', enabled=True):
                        # Forward passes
                        fake_B, _ = G_A2B(real_A)      # Human to Anime
                        fake_A, _ = G_B2A(real_B)      # Anime to Human
                        rec_A, _ = G_B2A(fake_B)       # Reconstruct Human
                        rec_B, _ = G_A2B(fake_A)       # Reconstruct Anime
                        
                        pred_fake_B = D_B(fake_B)
                        pred_fake_A = D_A(fake_A)

                        # GAN loss
                        loss_G_gan = generator_hinge_loss(pred_fake_B) + generator_hinge_loss(pred_fake_A)

                        # Cycle-consistency loss
                        loss_cycle = (criterion_cycle(rec_A, real_A) + criterion_cycle(rec_B, real_B)) * args.cycle_w

                        # Identity loss
                        id_loss = (criterion_identity(G_B2A(real_A)[0], real_A) + criterion_identity(G_A2B(real_B)[0], real_B)) * args.id_w

                        # Perceptual loss
                        perc_loss = 0.0
                        if perceptual:
                            perc_loss = perceptual(fake_B, real_B) * args.perc_w + perceptual(fake_A, real_A) * args.perc_w

                        # Total generator loss
                        loss_G_total = (loss_G_gan + loss_cycle + id_loss + perc_loss) / args.accum_steps

                    scaler.scale(loss_G_total).backward()
                    if (step+1) % args.accum_steps == 0 or (step+1)==steps_per_epoch:
                        scaler.step(optimizer_G)
                        scaler.update()
                # If scaler is None: Standard full-precision training (no mixed precision), computes losses and updates generator
                else:
                    optimizer_G.zero_grad(set_to_none=True)
                    fake_B, _ = G_A2B(real_A)
                    fake_A, _ = G_B2A(real_B)
                    rec_A, _ = G_B2A(fake_B)
                    rec_B, _ = G_A2B(fake_A)
                    loss_G_gan = generator_hinge_loss(D_B(fake_B)) + generator_hinge_loss(D_A(fake_A))
                    loss_cycle = (criterion_cycle(rec_A, real_A) + criterion_cycle(rec_B, real_B)) * args.cycle_w
                    id_loss = (criterion_identity(G_B2A(real_A)[0], real_A) + criterion_identity(G_A2B(real_B)[0], real_B)) * args.id_w
                    perc_loss = 0.0
                    if perceptual:
                        perc_loss = perceptual(fake_B, real_B) * args.perc_w + perceptual(fake_A, real_A) * args.perc_w
                    (loss_G_gan + loss_cycle + id_loss + perc_loss).backward()
                    optimizer_G.step()

            # Train Discriminators
            if scaler:
                optimizer_D.zero_grad(set_to_none=True)
                with autocast(device_type='cuda', enabled=True):
                    # Add small Gaussian noise to real/fake images to improve stability
                    noise_real_A = real_A + 0.01*torch.randn_like(real_A)
                    noise_real_B = real_B + 0.01*torch.randn_like(real_B)
                    fake_A_buffer = replay_buffer_A.push_and_pop(fake_A.detach())
                    fake_B_buffer = replay_buffer_B.push_and_pop(fake_B.detach())

                    # Hinge loss for discriminators
                    loss_D = (discriminator_hinge_loss(D_A(noise_real_A), D_A(fake_A_buffer)) +
                              discriminator_hinge_loss(D_B(noise_real_B), D_B(fake_B_buffer))) / args.accum_steps
                scaler.scale(loss_D).backward()
                if (step+1) % args.accum_steps == 0 or (step+1)==steps_per_epoch:
                    scaler.step(optimizer_D)
                    scaler.update()
            # If scaler is None: Standard full-precision training (no mixed precision), computes losses and updates discriminator
            else:
                optimizer_D.zero_grad(set_to_none=True)
                fake_A_buffer = replay_buffer_A.push_and_pop(fake_A.detach())
                fake_B_buffer = replay_buffer_B.push_and_pop(fake_B.detach())
                loss_D = (discriminator_hinge_loss(D_A(real_A), D_A(fake_A_buffer)) +
                          discriminator_hinge_loss(D_B(real_B), D_B(fake_B_buffer))) / args.accum_steps
                loss_D.backward()
                optimizer_D.step()

            # Track epoch losses
            epoch_G_loss += float(loss_G_total.detach().cpu())
            epoch_D_loss += float(loss_D.detach().cpu())
            steps += 1
            loop.set_postfix(G=f"{epoch_G_loss/steps:.4f}", D=f"{epoch_D_loss/steps:.4f}")

        # 12. Update learning rates
        scheduler_G.step()
        scheduler_D.step()

        # 13. Save sample outputs every 5 epochs
        if epoch % 5 == 0:
            save_samples(epoch, G_A2B, dataloader, device, args.output_dir)

        # 14. Save checkpoint every save_freq epochs
        if epoch % args.save_freq==0 or epoch==args.epochs:
            torch.save({
                'epoch': epoch, 'G_A2B': G_A2B.state_dict(), 'G_B2A': G_B2A.state_dict(),
                'D_A': D_A.state_dict(), 'D_B': D_B.state_dict(),
                'opt_G': optimizer_G.state_dict(), 'opt_D': optimizer_D.state_dict(),
            }, latest_checkpoint)
            tqdm.write(f'Checkpoint saved at epoch {epoch}')

        tqdm.write(f'Epoch {epoch} summary — G_loss: {epoch_G_loss/steps:.4f}, D_loss: {epoch_D_loss/steps:.4f}')

    tqdm.write('Training finished.')

if __name__ == '__main__':
    try:
        import multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser(description='Max-quality UGATIT training (512px)')
    parser.add_argument('--human_dir', type=str, required=True)
    parser.add_argument('--anime_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--bs', type=int, default=2, help='per-device batch size')
    parser.add_argument('--accum_steps', type=int, default=2, help='gradient accumulation steps')
    parser.add_argument('--lr_G', type=float, default=2e-4, help='learning rate for generator')
    parser.add_argument('--lr_D', type=float, default=1e-4, help='learning rate for discriminator')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--img_size', type=int, default=256)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--output_dir', type=str, default='outputs')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--save_freq', type=int, default=5)
    parser.add_argument('--id_w', type=float, default=15.0, help='identity loss weight')
    parser.add_argument('--cycle_w', type=float, default=10.0, help='cycle loss weight')
    parser.add_argument('--perc_w', type=float, default=1.0, help='perceptual loss weight')
    parser.add_argument('--id_loss_freq', type=int, default=10)
    parser.add_argument('--use_perceptual', action='store_true', default=True)
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--lr_decay_start', type=int, default=100)
    parser.add_argument('--max_steps_per_epoch', type=int, default=2000,
                        help='Maximum number of training steps (batches) per epoch. Use smaller number to speed up.')

    args = parser.parse_args()

    # basic checks
    assert os.path.isdir(args.human_dir), f"human_dir not found: {args.human_dir}"
    assert os.path.isdir(args.anime_dir), f"anime_dir not found: {args.anime_dir}"

    train_model(args)
