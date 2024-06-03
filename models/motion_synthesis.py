import torch
import numpy as np
from tqdm import tqdm
from diffusers import UNet2DModel, VQModel, DDPMScheduler

class VQModel_(VQModel):
    
    @torch.no_grad()
    def encode_frame(self, frame):
        return self.encode(frame).latents

    @torch.no_grad()
    def encode_spec(self, spec, batch_size=12):
        b, f, c, h, w = spec.shape
        spec = spec.reshape(-1, h, w)
        out = []
        for i in range(int(np.ceil(spec.shape[0] / batch_size))):
            inp = spec[i * batch_size:(i + 1) * batch_size, None, :, :].repeat(1, 3, 1, 1)
            out.append(self.encode(inp).latents) # b',3,h/4,w/4
        out = torch.cat(out, dim=0)
        return out.reshape(b, f, -1, *out.shape[-2:])  # b,f,c*3,h/4,w/4
    
    @torch.no_grad()
    def decode_spec(self, spec, batch_size=12):
        b, f, _, h, w = spec.shape
        spec = spec.reshape(-1, 3, h, w)
        out = []
        for i in range(int(np.ceil(spec.shape[0] / batch_size))):
            inp = spec[i * batch_size:(i + 1) * batch_size, :, :, :]
            out.append(self.decode(inp).sample.mean(dim=1)) # b',h*4,w*4
        out = torch.cat(out, dim=0)
        return out.reshape(b, f, -1, *out.shape[-2:])  # b,f,c,h*4,w*4

def get_pretrained_vae():
    return VQModel_.from_pretrained("CompVis/ldm-celebahq-256", subfolder="vqvae")

def get_noise_scheduler():
    return DDPMScheduler(num_train_timesteps=1000, beta_start=0.0015, beta_end=0.0195, beta_schedule="scaled_linear")

@torch.no_grad()
def generate_spectrum(
    vae, 
    unet, 
    noise_scheduler, 
    frame: torch.Tensor,
    num_freq: int = 16, 
    num_steps: int = 1000, 
    batch_size: int = 1,
):
    assert isinstance(frame, torch.Tensor) and len(frame.shape) == 4 and frame.shape[0] == 1, (type(frame), frame.shape)
    
    frame = vae.encode_frame(frame)
    freq_idx = torch.arange(num_freq, dtype=torch.long, device=frame.device)
    noise_scheduler.set_timesteps(num_steps, device=frame.device)
    spec = []
    nb = int(np.ceil(num_freq / batch_size))
    for i in tqdm(range(nb)):
        j = i * batch_size
        bs = min(num_freq, j + batch_size) - j
        sample = torch.randn(bs, unet.out_channels, *frame.shape[2:], dtype=frame.dtype, device=frame.device)
        for t in noise_scheduler.timesteps:
            noise_pred = unet(torch.cat([sample, frame.repeat(bs, 1, 1, 1)], dim=1), timestep=t, class_labels=freq_idx[j:j + batch_size]).sample
            sample = noise_scheduler.step(noise_pred, t, sample).prev_sample
        sample = vae.decode_spec(sample.unsqueeze(0)).squeeze(0)
        spec.append(sample.cpu())
    spec = torch.cat(spec, dim=0)
    
    return spec.permute(0, 2, 3, 1).numpy()
