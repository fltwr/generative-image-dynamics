import os
import datetime
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
import torch
import torch.nn.functional as F
from diffusers import UNet2DModel
from utils import *
from models.motion_synthesis import *

if torch.cuda.is_available():
    DEVICE = torch.device("cuda:0")
else:
    DEVICE = torch.device("cpu")
FFT = True
NUM_FREQ = 16
SPEC_CHANNELS = 4 if FFT else 2
FRAME_CHANNELS = 3
_VAE_LATENT_CHANNELS = 3
SPEC_LATENT_CHANNELS = SPEC_CHANNELS * _VAE_LATENT_CHANNELS
FRAME_LATENT_CHANNELS = _VAE_LATENT_CHANNELS
LATENT_HEIGHT = 40
LATENT_WIDTH = 64
HEIGHT = LATENT_HEIGHT * 4
WIDTH = LATENT_WIDTH * 4
_NAME = "unet_v1"
_MODEL_DIR = "data/models"
if not os.path.exists(_MODEL_DIR):
    os.makedirs(_MODEL_DIR)
CKPT_PATH = os.path.join(_MODEL_DIR, _NAME + ".pth")
LOSS_PATH = os.path.join(_MODEL_DIR, _NAME + "_loss.png")
OUT_DIR = os.path.join(_MODEL_DIR, _NAME + "_samples")
BASE_LR = 1e-5
BATCH_SIZE = 1
NUM_WORKERS = 0

# pretrained VAE
vae = get_pretrained_vae().to(DEVICE).eval()

# noise scheduler
noise_scheduler = get_noise_scheduler()

# unet
model = UNet2DModel(**{
    "in_channels": SPEC_LATENT_CHANNELS + FRAME_LATENT_CHANNELS,
    "out_channels": SPEC_LATENT_CHANNELS,
    "class_embed_type": "timestep",
}).to(DEVICE)

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=BASE_LR * BATCH_SIZE)

# data
train_loader = torch.utils.data.DataLoader(FrameSpectrumDataset(NUM_FREQ, is_train=True, fft=FFT), BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
print(len(train_loader), "train batches")

def plot_loss(losses, show_plot=False):
    m = len(train_loader)

    y = np.array(losses["train_loss"])
    x = np.arange(y.shape[0], dtype=y.dtype)
    x /= m
    plt.plot(x, np.log(y), label="train loss", alpha=.5, linewidth=.05)

    n = y.shape[0] // m
    y = y[:n * m].reshape((n, m)).mean(axis=1)
    x = np.arange(n, dtype=y.dtype) + 0.5
    plt.plot(x, np.log(y), label="train loss (epoch mean)", alpha=.8)

    plt.xlabel(f"Epoch")
    plt.ylabel("Log loss")
    plt.legend()
    
    plt.savefig(LOSS_PATH, bbox_inches="tight")
    print(f"plot saved at {LOSS_PATH}")
    if show_plot:
        plt.show()
    plt.close()

def train_iteration(model, batch):
    model.train()

    frame, freq_idx, spec = batch

    spec = vae.encode_spec(spec.unsqueeze(1).to(DEVICE)).squeeze(1)
    noise = torch.randn_like(spec)
    timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (spec.shape[0],), dtype=torch.long, device=DEVICE)
    spec = noise_scheduler.add_noise(spec, noise, timesteps)
    frame = vae.encode_frame(frame.to(DEVICE))
    noise_pred = model(torch.cat([spec, frame], dim=1), timesteps, freq_idx.to(DEVICE), return_dict=False)[0]

    return F.mse_loss(noise_pred, noise, reduction="mean")

@torch.no_grad()
def test(model, test_ids=None, num_freq=16, num_steps=100, batch_size=1):
    assert num_freq <= NUM_FREQ, (num_freq, NUM_FREQ)

    model.eval()
    
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)

    if test_ids is None:
        test_ids = ["512px-Dandelion_picture.jpg"]
    elif isinstance(test_ids, str):
        test_ids = [test_ids]

    for tid in test_ids:
        frame_np = get_image(f"data/images/{tid}", WIDTH, HEIGHT, crop=True)
        frame = train_loader.dataset.process_frame(frame_np).unsqueeze(0).to(DEVICE)
        
        spec_np = generate_spectrum(vae, model, noise_scheduler, frame, num_freq=num_freq, num_steps=num_steps, batch_size=batch_size)
        
        spec_image, video = visualize_sample(frame_np, spec_np, train_loader.dataset, magnification=2.0, include_flow=True, fps=30)
        
        ts = datetime.datetime.now().isoformat().replace(":", "_")
        spec_image.save(os.path.join(OUT_DIR, f"{tid}_{ts}_ddpm{num_steps}_spec.png"))
        video.write_videofile(os.path.join(OUT_DIR, f"{tid}_{ts}_ddpm{num_steps}_flow.mp4"), logger=None)

if __name__ == "__main__":
    training = Training(model, optimizer, lr_scheduler=None, ckpt_path=CKPT_PATH)    
    training.run(
        max_niters=len(train_loader) * 300,
        train_loader=train_loader,
        train_iteration=train_iteration,
        test=test,
        plot_loss=plot_loss,
        print_step=10,
        plot_step=len(train_loader),
        save_step=len(train_loader),
        test_step=len(train_loader) * 5,
    )
