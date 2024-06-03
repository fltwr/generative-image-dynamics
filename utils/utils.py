import cv2
import json
import torch
import scipy
import subprocess
import numpy as np
from typing import Union
from PIL import Image
import moviepy.editor as mpy
from .flow_vis import make_colorwheel
DTYPE_NUMPY =  np.float32
DTYPE_TORCH = torch.float32

def flow_to_spec(flow: np.ndarray, fft: bool = True):
    assert len(flow.shape) == 4, flow.shape
    if fft:
        spec = np.fft.fft(flow, axis=0)
        return np.concatenate([spec.real, spec.imag], axis=-1)
    else:
        return scipy.fft.dct(flow, axis=0, norm="ortho", orthogonalize=True)

def spec_to_flow(spec: np.ndarray, fft: bool = True):
    ndims = len(spec.shape)
    assert ndims in [4, 5], spec.shape
    axis = 0 if ndims == 4 else 1
    if fft:
        assert spec.shape[-1] == 4, spec.shape
        return np.fft.ifft(spec[..., :2] + spec[..., 2:] * 1j, axis=axis)
    else:
        assert spec.shape[-1] == 2, spec.shape
        return scipy.fft.idct(spec, axis=axis, norm="ortho", orthogonalize=True)

def truncate_spectrum(spec: np.ndarray, num_freq: int, fft: None = None):
    assert len(spec.shape) == 2 or len(spec.shape) == 4, spec.shape
    f, *hw, c = spec.shape
    assert 0 < num_freq <= f, (num_freq, f)
    return spec[:num_freq]

def pad_spectrum(spec: np.ndarray, num_freq_total: int, fft: bool = True):
    assert len(spec.shape) in [4, 5], spec.shape
    *b, f, h, w, c = spec.shape
    if fft:
        num_pos_freq = (num_freq_total + 1) // 2
        assert 0 < f <= num_pos_freq, (num_freq_total, num_pos_freq, f)
    
    padded = np.zeros((*b, num_freq_total, h, w, c), dtype=spec.dtype)
    padded[..., :f, :, :, :] = spec
    
    if fft and f > 1:
        # coefficients for negative frequencies by conjugate symmetry
        padded[..., -1:-f:-1, :, :, :2] = spec[..., 1:, :, :, :2]
        padded[..., -1:-f:-1, :, :, 2:] = -spec[..., 1:, :, :, 2:]
    
    return padded

def normalize_spectrum(spec: np.ndarray, std: np.ndarray, scale: float):
    assert len(spec.shape) == len(std.shape) == 4, (spec.shape, std.shape)
    assert spec.shape[0] == std.shape[0], (spec.shape, std.shape)  # same number of frequencies
    # return np.clip(spec / std / scale, -1, 1)
    return spec / std / scale

def denormalize_spectrum(spec, std, scale: float):
    assert len(spec.shape) in [4, 5], spec.shape
    assert len(spec.shape) == len(std.shape), (spec.shape, std.shape)
    return spec * std * scale

class FrameProcessing:

    def process_frame(self, frame):
        # Input : image, numpy.ndarray, shape (height, width, 3), range [0, 255]
        # Output: image, torch.Tensor, shape (3, height, width), range [-1, 1]
        frame = frame.astype(DTYPE_NUMPY) / 127.5 - 1
        return torch.from_numpy(frame).permute(2, 0, 1)
    
    def deprocess_frame(self, frame):
        # Input : image, torch.Tensor, shape (batch_size, 3, height, width), range [-1, 1]
        # Output: image, numpy.ndarray, shape (batch_size, height, width, 3), range [0, 255]
        frame = (frame.permute(0, 2, 3, 1).numpy() + 1) * 127.5
        return np.clip(frame, 0, 255).astype(np.uint8)

class FrameSpectrumProcessing(FrameProcessing):
    
    def __init__(
        self,
        num_freq: int,
        fft: bool = True,
        scale: float = 2.82,
        std_path: str = "data/labels/fft_std.npy",
    ):
        assert scale > 0, scale
        self.num_freq = num_freq
        
        self.scale = scale
        self.std = load_npy(std_path)[:, None, None, None]
        
        self.fft = fft
        self.num_channels = 4 if fft else 2
        self.num_freq_total = self.std.shape[0]
        
    def process_spec(self, spec):
        # Input : spectrum, numpy.ndarray, shape (num_frequencies, height, width, num_channels), unnormalized
        # Output: spectrum, torch.Tensor, shape (num_frequencies, num_channels, height, width), range [-1, 1]
        assert len(spec.shape) == 4 and spec.shape[3] == self.num_channels, (spec.shape, self.num_channels)
        spec = normalize_spectrum(spec[:self.num_freq], self.std[:self.num_freq], self.scale)
        # spec = np.sqrt(np.abs(spec)) * np.sign(spec)
        return torch.from_numpy(spec.astype(DTYPE_NUMPY)).permute(0, 3, 1, 2)
    
    def deprocess_spec(self, spec):
        # Input : spectrum, torch.Tensor, shape (batch_size, num_frequencies, num_channels, height, width), range [-1, 1]
        # Output: optical flow, numpy.ndarray, shape (batch_size, num_frames, height, width, 2)
        assert len(spec.shape) == 5 and spec.shape[2] == self.num_channels, (spec.shape, self.num_channels)
        spec = spec.permute(0, 1, 3, 4, 2).numpy()
        # spec = np.square(spec) * np.sign(spec)
        spec = denormalize_spectrum(spec, self.std[None, :spec.shape[1]], self.scale)
        flow = spec_to_flow(pad_spectrum(spec, self.num_freq_total, fft=self.fft), fft=self.fft)
        return spec, flow.real.astype(DTYPE_NUMPY)

class FrameFlowProcessing:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=DTYPE_TORCH)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=DTYPE_TORCH)
    
    def __init__(self, frame_h=160, frame_w=256):
        self.frame_h = frame_h
        self.frame_w = frame_w
    
    def get_frames(self, video_path, start_sec, num_frames, fps):
        frames = get_frames(video_path, self.frame_w, self.frame_h, start_sec, t=None, f=num_frames, fps=fps)
        assert frames.shape[0] == num_frames, (frames.shape[0], num_frames)
        return frames
    
    def process_frame(self, frame):
        ndims = len(frame.shape)
        assert ndims in [3, 4], frame.shape
        frame = torch.from_numpy(frame.astype(DTYPE_NUMPY)) / 255
        if ndims == 3:  # a single frame
            return (frame.permute(2, 0, 1) - self.mean[:, None, None]) / self.std[:, None, None]
        else:  # multiple frames
            return (frame.permute(0, 3, 1, 2) - self.mean[None, :, None, None]) / self.std[None, :, None, None]
    
    def denormalize_frame(self, frame):
        assert len(frame.shape) == 4, frame.shape
        return torch.clip(frame * self.std[None, :, None, None].to(frame.device) + self.mean[None, :, None, None].to(frame.device), 0, 1)

    def deprocess_frame(self, frame):
        assert len(frame.shape) == 4, frame.shape
        frame = frame.cpu() * self.std[None, :, None, None] + self.mean[None, :, None, None]
        return np.clip(frame.permute(0, 2, 3, 1).numpy() * 255, 0, 255).astype(np.uint8)
    
    def process_flow(self, flow):
        ndims = len(flow.shape)
        assert ndims in [3, 4], flow.shape
        flow = torch.from_numpy(flow)
        if ndims == 3:  # a single flow field
            return flow.permute(2, 0, 1)
        else:  # multiple flow fields
            return flow.permute(0, 3, 1, 2)
    
    def deprocess_flow(self, flow):
        assert len(flow.shape) == 4, flow.shape
        return flow.cpu().permute(0, 2, 3, 1).numpy()

def _cartesian_to_polar(x, y):
    rad = np.sqrt(np.square(x) + np.square(y))
    ang = np.arctan2(-y, -x)
    return rad, ang

def _radius_angle_to_color_video(rad, ang, rad_max=None):
    """Modified from flow_vis (https://github.com/tomrunia/OpticalFlow_Visualization)."""
    assert len(rad.shape) == len(ang.shape) == 3, (rad.shape, ang.shape)  # thw
    assert rad.shape == ang.shape, (rad.shape, ang.shape)
    
    if rad_max is None:
        rad_max = rad.max()
    epsilon = 1e-5
    rad = rad / (rad_max + epsilon)
    ang = ang / np.pi
    
    color_video = np.zeros((*rad.shape, 3), np.uint8)
    
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    
    fk = (ang + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        color_video[:, :, :, i] = np.floor(255 * col)
    return color_video

def flow_to_color_video(flow: np.ndarray, rad_max: Union[float, None] = None):
    assert len(flow.shape) == 4, flow.shape
    rad, ang = _cartesian_to_polar(flow[..., 0], flow[..., 1])
    return _radius_angle_to_color_video(rad, ang, rad_max=rad_max)

def spectrum_to_color_video(spec: np.ndarray, fft_axis="real"):
    # note the input spectrum is assumed to be normalized (range [-1 , 1])
    assert len(spec.shape) == 4 and spec.shape[3] in [2, 4] and np.isreal(spec).all(), spec.shape
    
    rad_max = 1
    if spec.shape[3] == 2:
        return flow_to_color_video(spec, rad_max=rad_max)
    else:
        assert fft_axis in ["real", "imag", "angle", "magnitude"]
        if fft_axis == "real":
            return flow_to_color_video(spec[..., :2], rad_max=rad_max)
        
        elif fft_axis == "imag":
            return flow_to_color_video(spec[..., 2:], rad_max=rad_max)
        
        elif fft_axis == "angle":
            ang = np.angle(spec[..., :2] + spec[..., 2:] * 1j) / np.pi
            return flow_to_color_video(ang, rad_max=rad_max)
        
        elif fft_axis == "magnitude":
            uv = np.sqrt(np.square(spec[..., :2]) + np.square(spec[..., 2:]))
            rad, ang = _cartesian_to_polar(uv[..., 0], uv[..., 1])
            ang = ang * 4 + np.pi * 3  # rescale from [-pi, -pi/2] to [-pi, pi]
            return _radius_angle_to_color_video(rad, ang, rad_max=rad_max)

def _pad_video(video, l, r, t, b, color):
    canvas = np.zeros((video.shape[0], video.shape[1] + t + b, video.shape[2] + l + r, video.shape[3]), dtype=video.dtype)
    canvas[...] = color
    canvas[:, t:t + video.shape[1], l:l + video.shape[2], :] = video
    return canvas

def add_video_title(video, text, height=24, bg_color=(255, 255, 255), text_color=(0, 0, 0)):
    title = np.zeros((height, video.shape[2], 3), dtype=np.uint8)
    title[...] = bg_color
    title = cv2.putText(
        title,
        text,
        (8, height - 8),           # location
        cv2.FONT_HERSHEY_SIMPLEX,  # font
        .45,                       # font size
        text_color,
        1,                         # thickness
    )
    video = np.concatenate([title[None, ...].repeat(video.shape[0], axis=0), video], axis=1)
    return _pad_video(video, 2, 2, 0, 2, bg_color)

def visualize_sample(frame_np: np.ndarray, spec_np: np.ndarray, transforms: FrameSpectrumProcessing, magnification=5.0, include_flow=True, fps=30):
    assert len(frame_np.shape) == 3, frame_np.shape
    assert len(spec_np.shape) == 4, spec_np.shape
    
    # show spectrum in an image
    spec_image = np.concatenate([
        np.concatenate([
            frame_np, 
            np.full_like(frame_np, 255),
        ], axis=1),
        np.concatenate([
            spectrum_to_color_video(spec_np, fft_axis="real"),
            spectrum_to_color_video(spec_np, fft_axis="imag"),
        ], axis=2).reshape((spec_np.shape[0] * spec_np.shape[1], -1, 3)),
    ], axis=0)
    
    # post-process spectrum and compute flow
    d = np.clip(np.median(np.abs(spec_np), axis=(1, 2)) - .05, a_min=0, a_max=None)
    d = np.minimum(np.abs(spec_np), d[:, None, None, :]) * np.sign(spec_np)
    _, flow = transforms.deprocess_spec(torch.from_numpy(spec_np - d).permute(0, 3, 1, 2).unsqueeze(0))
    flow = flow[0]
    flow[..., 0] *= magnification
    flow[..., 1] *= magnification * frame_np.shape[0] / frame_np.shape[1]
    
    # show warped frames
    video = remap(frame_np, flow, cv2.INTER_CUBIC, include_first_frame=False)
    video = add_video_title(video, "Warped frames")
    if include_flow:
        # show optical flow on the right side
        flow_color = add_video_title(flow_to_color_video(flow, rad_max=None), "Synthesized optical flow")
        video = np.concatenate([flow_color, video], axis=2)
    
    return Image.fromarray(spec_image), mpy.ImageSequenceClip([video[i] for i in range(video.shape[0])], fps=fps)

def remap(frame, flow, interpolation=cv2.INTER_CUBIC, include_first_frame=True):
    assert len(frame.shape) == 3 and len(flow.shape) == 4 and frame.shape[:2] == flow.shape[1:3], (frame.shape, flow.shape)
    h, w = frame.shape[:2]
    
    # interpolation = interpolation + cv2.WARP_RELATIVE_MAP
    x = np.arange(w)[None, :].repeat(h, axis=0).astype(flow.dtype)
    y = np.arange(h)[:, None].repeat(w, axis=1).astype(flow.dtype)
    
    frames = []
    if include_first_frame:
        frames.append(frame)
    for i in range(flow.shape[0]):
        frames.append(cv2.remap(frame, x - flow[i, ..., 0], y - flow[i, ..., 1], interpolation, borderMode=cv2.BORDER_REPLICATE))
    frames = np.stack(frames, axis=0)
    
    return frames

def _get_video_stream(video_path: str):
    args = ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path]
    try:
        output = subprocess.check_output(args, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise Exception(e.output.decode("utf-8"))

    output = output.decode("utf-8")
    output = json.loads(output)
    output = output["streams"]

    video_stream = next((stream for stream in output if stream["codec_type"] == "video"), None)
    if video_stream is None:
        raise Exception(f"{video_path}: video stream not found")

    return video_stream

def get_video_duration(video_path: str, copy: bool = False):
    video_stream = _get_video_stream(video_path)
    duration = video_stream.get("duration")
    if duration is not None:
        return float(duration)
    
    c = ["-c", "copy"] if copy else []
    args = ["ffmpeg", "-i", video_path, "-map", "0:v:0", *c, "-f", "null", "-"]
    try:
        output = subprocess.check_output(args, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        raise Exception(e.output.decode("utf-8"))
    output = output.decode("utf-8")
    output = output.splitlines()
    for line in output[::-1]:
        if line.startswith("frame="):
            break
    else:
        line = output[-1]
    # frame_num = int(line.split("frame=")[1].split("fps=")[0]) if "frame=" in line else None
    hh, mm, ss = line.split("time=")[1].split("bitrate=")[0].strip().split(":")
    duration = int(hh) * 3600 + int(mm) * 60 + float(ss)
    return duration

def get_frames(inp: str, w: int, h: int, start_sec: float = 0, t: float = None, f: int = None, fps = None) -> np.ndarray:
    args = []
    if t is not None:
        args += ["-t", f"{t:.2f}"]
    elif f is not None:
        args += ["-frames:v", str(f)]
    if fps is not None:
        args += ["-r", str(fps)]
    
    args = ["ffmpeg", "-nostdin", "-ss", f"{start_sec:.2f}", "-i", inp, *args, 
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "pipe:"]
    
    process = subprocess.Popen(args, stderr=-1, stdout=-1)
    out, err = process.communicate()
    retcode = process.poll()
    if retcode:
        raise Exception(f"{inp}: ffmpeg error: {err.decode('utf-8')}")

    return np.frombuffer(out, np.uint8).reshape(-1, h, w, 3)

def trim_video(inp_file: str, start_sec: float, t: float, out_file: str):
    args = ["ffmpeg", "-ss", f"{start_sec:.2f}", "-i", inp_file, "-t", f"{t:.2f}", out_file]

    process = subprocess.Popen(args, stderr=subprocess.STDOUT, stdout=subprocess.PIPE)
    out, err = process.communicate()
    retcode = process.poll()
    if retcode:
        raise Exception(f"{out_file}: ffmpeg error: {out.decode('utf-8')}")

def save_video(frames, fps, filename):
    nframes = frames.shape[0]
    clip = mpy.ImageSequenceClip([frames[i] for i in range(nframes)], fps=fps)
    clip.write_videofile(filename, fps=fps, logger=None)
    
def save_gif(frames, fps, filename):
    nframes = frames.shape[0]
    clip = mpy.ImageSequenceClip([frames[i] for i in range(nframes)], fps=fps)
    clip.write_gif(filename, program="ffmpeg", fps=fps, logger=None)  # fps doesn't work with program="ImageMagick"

def get_image(filename, width=None, height=None, crop=False):
    im = Image.open(filename)
    if width is not None and height is not None and im.size != (width, height):
        w0, h0 = im.size
        if crop:
            ratio = max(width / w0, height / h0)
            w1 = max(int(ratio * w0), width)
            h1 = max(int(ratio * h0), height)
            l = (w1 - width) // 2
            t = (h1 - height) // 2
            im = im.resize((w1, h1)).crop((l, t, l + width, t + height))
        else:
            im = im.resize((width, height))
    return np.array(im)

def save_image(image, filename):
    Image.fromarray(image).save(filename)

def save_npy(arr, filename, dtype=DTYPE_NUMPY):
    with open(filename, "wb") as f:
        np.save(f, arr.astype(dtype))

def load_npy(filename):
    with open(filename, "rb") as f:
        return np.load(f).astype(DTYPE_NUMPY)
