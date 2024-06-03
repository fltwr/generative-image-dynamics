import os
import torch
import random
import numpy as np
from tqdm import tqdm
from .utils import *

class SpectrumDataset(torch.utils.data.Dataset, FrameSpectrumProcessing):
    """Dataset for motion synthesis VAE."""
    
    def __init__(
        self, 
        num_freq: int,
        is_train: bool, 
        fft: bool = True,
        scale: float = 2.82,
        std_path: str = "data/labels/fft_std.npy",
        label_dir: str = "data/labels",
        video_dir: str = "data/videos",
        flow_dir: str = "data/flow",
    ):
        super().__init__(num_freq, fft, scale, std_path)
        
        self.is_train = is_train
        self.video_dir = video_dir
        self.flow_dir = flow_dir
        self.label_dir = label_dir
        
        with open(os.path.join(label_dir, f"motion_synthesis_{'train' if is_train else 'test'}_set.csv")) as f:
            self.data = {}
            num_seqs = 0
            
            f.readline()
            for line in f:
                video_id, start_sec, num_frames, fps = line.strip().split(",")
                start_sec, num_frames, fps = int(start_sec), int(num_frames), float(fps)
                flow_path = os.path.join(flow_dir, f"{video_id}_{start_sec:03d}.npy")
                if os.path.exists(flow_path):
                    if video_id not in self.data:
                        self.data[video_id] = []
                    self.data[video_id].append({"start_sec": start_sec, "fps": fps})
                    num_seqs += 1
            self.data = list(self.data.items())
        
        print(f"{type(self).__name__} ({'train' if self.is_train else 'test'}): {len(self.data)} videos, {num_seqs} sequences")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        video_id, sequences = self.data[idx]
        if self.is_train:
            seq = random.choice(sequences)
        else:
            seq = sequences[0]
        spec = self._get_spec(video_id, seq["start_sec"])
        if self.is_train:
            i = random.choice(range(self.num_freq))
            return spec[i]  # shape (num_channels, height, width)
        else:
            return spec  # shape (num_frequencies, num_channels, height, width)
    
    def _get_spec(self, video_id, start_sec):
        flow = load_npy(os.path.join(self.flow_dir, f"{video_id}_{start_sec:03d}.npy"))
        spec = flow_to_spec(flow, fft=self.fft)
        return self.process_spec(spec)
    
    def get_std_from_zero(self, max_seqs_per_vid=1):
        """Compute standard deviation from zero of motion spectrums, which is needed to normalize input data to the diffusion model."""
        import matplotlib.pyplot as plt
        plt.style.use("seaborn-v0_8-whitegrid")
        
        std = np.zeros((149,), dtype=DTYPE_NUMPY)
        for vid, seqs in tqdm(self.data):
            n = 0
            m = np.zeros_like(std)
            for x in seqs:
                flow = load_npy(os.path.join(self.flow_dir, f"{vid}_{x['start_sec']:03d}.npy"))
                spec = flow_to_spec(flow, fft=self.fft)
                m += np.square(spec).mean(axis=(1, 2, 3))
                n += 1
                if max_seqs_per_vid is not None and n >= max_seqs_per_vid:
                    break
            std += m / n
        std = np.sqrt(std / len(self.data))
        
        plt.plot(std)
        plt.xlabel("Frequency index")
        plt.ylabel("Standard deviation")
        plt.title(f"{'FFT' if self.fft else 'DCT'} standard deviation from zero")
        plt.show()
        plt.close()
        
        return std
    
    def test_scales(self, std, min_scale=1, max_scale=4, num_scales=7, max_seqs_per_vid=1):
        """Compute percentage of values in spectrums that is out of the range [-1, 1] with the given `std` and different values of `scale`."""
        import matplotlib.pyplot as plt
        plt.style.use("seaborn-v0_8-whitegrid")
        
        std = std[:self.num_freq, None, None, None]
        scales = np.linspace(min_scale, max_scale, num_scales)

        out_of_range = np.zeros((self.num_freq, num_scales), dtype=DTYPE_NUMPY)
        for vid, seqs in tqdm(self.data):
            n = 0
            m = np.zeros_like(out_of_range)
            for x in seqs:
                flow = load_npy(os.path.join(self.flow_dir, f"{vid}_{x['start_sec']:03d}.npy"))
                spec = flow_to_spec(flow, fft=self.fft)[:self.num_freq] / std
                spec = np.abs(spec)
                for i, s in enumerate(scales):
                    m[:, i] += ((spec / s) > 1).astype(spec.dtype).mean(axis=(1, 2, 3))
                n += 1
                if max_seqs_per_vid is not None and n >= max_seqs_per_vid:
                    break
            out_of_range += m / n
        out_of_range /= len(self.data)
        out_of_range *= 100
        
        for i in range(self.num_freq):
            plt.plot(scales, out_of_range[i, :], label=str(i), ls="-", alpha=.3)
        plt.plot(scales, out_of_range.mean(axis=0), label="mean", ls="-", marker="^")
        plt.xlabel("Scale")
        plt.ylabel("Percentage (%)")
        plt.title(f"Out of range values in {'FFT' if self.fft else 'DCT'}")
        plt.legend()
        
        return scales, out_of_range#.mean(axis=0)
    
    def reconstruct_flow(self, num_freqs=[1, 2, 4, 8, 16, 32, 64, 75]):
        """Compute mean squared errors of reconstructing optical flow from spectrums with limited numbers of frequencies."""
        import matplotlib.pyplot as plt
        plt.style.use("seaborn-v0_8-whitegrid")
        
        mse = np.zeros(len(num_freqs), dtype=DTYPE_NUMPY)
        for vid, seqs in tqdm(self.data):
            n = 0
            m = np.zeros_like(mse)
            for x in seqs:
                flow = load_npy(os.path.join(self.flow_dir, f"{vid}_{x['start_sec']:03d}.npy"))
                spec = flow_to_spec(flow, fft=self.fft)
                for i, num_freq in enumerate(num_freqs):
                    flow_ = spec_to_flow(pad_spectrum(truncate_spectrum(spec, num_freq, fft=self.fft), 149, fft=self.fft), fft=self.fft)
                    m[i] += np.sqrt(np.square(flow - flow_).mean())
                n += 1
                
                break
            
            mse += m / n
        mse /= len(self.data)
        
        plt.plot(num_freqs, mse)
        plt.xlabel("Number of frequences")
        plt.ylabel("Mean squared error")
        plt.title(f"Reconstructing optical flow from truncated {'FFT' if self.fft else 'DCT'}")
        plt.show()
        plt.close()
        
        return num_freqs, mse

class FrameSpectrumDataset(SpectrumDataset):
    """Dataset for motion synthesis U-Net."""
    
    def __getitem__(self, idx):
        video_id, sequences = self.data[idx]
        if self.is_train:
            seq = random.choice(sequences)
        else:
            seq = sequences[0]
        
        # motion spectrum
        spec = self._get_spec(video_id, seq["start_sec"])
        _, _, h, w = spec.shape
        if self.is_train:
            freq_idx = torch.randint(0, self.num_freq, tuple(), dtype=torch.long)
            spec = spec[freq_idx]
        else:
            freq_idx = torch.arange(self.num_freq, dtype=torch.long)
        
        # first frame
        frame = get_frames(os.path.join(self.video_dir, f"{video_id}.mp4"), w, h, seq["start_sec"], f=1, fps=seq["fps"])[0]

        return (self.process_frame(frame), freq_idx, spec)

class FrameFlowDataset(torch.utils.data.Dataset, FrameFlowProcessing):
    """Dataset for frame synthesis model."""
    
    def __init__(
        self, 
        is_train: bool,
        frame_h: int = 160, 
        frame_w: int = 256,  
        label_dir: str = "data/labels",
        video_dir: str = "data/videos",
        flow_dir: str = "data/flow",
    ):
        super().__init__(frame_h, frame_w)
        
        self.is_train = is_train
        self.video_dir = video_dir
        self.flow_dir = flow_dir
        
        with open(os.path.join(label_dir, f"frame_synthesis_{'train' if is_train else 'test'}_set.csv")) as f:
            self.data = []
            f.readline()
            for line in f:
                video_id, start_sec, num_frames, fps = line.strip().split(",")
                start_sec, num_frames, fps = int(start_sec), int(num_frames), float(fps)
                flow_path = os.path.join(flow_dir, f"{video_id}_{start_sec:03d}.npy")
                assert os.path.exists(flow_path), flow_path
                self.data.append((video_id, start_sec, num_frames, fps))
        
        print(f"{type(self).__name__} ({'train' if self.is_train else 'test'}): {len(self.data)} sequences")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        video_id, start_sec, num_frames, fps = self.data[idx]
        
        frames = self.get_frames(os.path.join(self.video_dir, f"{video_id}.mp4"), start_sec, num_frames, fps)
        flow = load_npy(os.path.join(self.flow_dir, f"{video_id}_{start_sec:03d}.npy"))
        
        if self.is_train:
            t = random.choice(range(1, num_frames))
            return (
                self.process_frame(frames[0]),   # source frame, shape (3, height, width)
                self.process_frame(frames[t]),   # target frame, shape (3, height, width)
                self.process_flow(flow[t - 1]),  # optical flow, shape (2, height, width)
            )
        else:
            return (
                self.process_frame(frames[0]),   # first frame, shape (3, height, width)
                self.process_frame(frames[1:]),  # other frames, shape (num_frames, 3, height, width)
                self.process_flow(flow),         # optical flow, shape (num_frames, 2, height, width)
            )
