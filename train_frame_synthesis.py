import os
import cv2
import torch
import datetime
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
from torchmetrics.image import PeakSignalNoiseRatio
from torchmetrics.image import StructuralSimilarityIndexMeasure
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from models.frame_synthesis import *
from utils import *

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_NAME = "frame_synthesis_v1"
_MODEL_DIR = "data/models"
if not os.path.exists(_MODEL_DIR):
    os.makedirs(_MODEL_DIR)
MODEL_PATH = os.path.join(_MODEL_DIR, _NAME + ".pth")
LOSS_PATH = os.path.join(_MODEL_DIR, _NAME + "_loss.png")
METRICS_PATH = os.path.join(_MODEL_DIR, _NAME + "_metrics.png")
OUT_DIR = os.path.join(_MODEL_DIR, _NAME + "_samples")
BASE_LR = 1e-5
BATCH_SIZE = 16
NUM_WORKERS = 0

model = Synthesis().to(DEVICE)
loss_fn = VGGLoss().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), BASE_LR * BATCH_SIZE)
psnr = PeakSignalNoiseRatio(data_range=1.0, reduction="elementwise_mean").to(DEVICE)
ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="elementwise_mean").to(DEVICE)
lpip = LearnedPerceptualImagePatchSimilarity(net_type="alex", reduction="mean", normalize=True).to(DEVICE)
print("LPIPS-" + lpip.net.net._get_name())

train_loader = torch.utils.data.DataLoader(FrameFlowDataset(is_train=True), BATCH_SIZE, num_workers=NUM_WORKERS, shuffle=True)
test_loader = torch.utils.data.DataLoader(FrameFlowDataset(is_train=False), 1, num_workers=NUM_WORKERS, shuffle=False)
print(f"{len(train_loader)} train batches, {len(test_loader)} test batches")

def plot_loss(losses, show_plot=False):
    # plot training loss
    plt.figure(figsize=(12, 3))
    y = np.array(losses["train_loss"])

    m = len(train_loader)
    n = y.shape[0] // m
    y = y[:n * m].reshape((n, m)).mean(axis=1)
    plt.plot(np.log(y), label="train loss")

    plt.xlabel("Epoch")
    plt.ylabel("Log loss")
    plt.title("Training loss")
    
    plt.savefig(LOSS_PATH, bbox_inches="tight")
    print(f"plot saved at {LOSS_PATH}")
    if show_plot:
        plt.show()
    plt.close()
    
    # plot evaluation metrics
    scores = defaultdict(list)
    iters = defaultdict(list)
    for ite in sorted(losses["evaluation"].keys()):
        for k, v in losses["evaluation"][ite].items():
            iters[k].append(ite)
            scores[k].append(v)
    epochs = {k: np.array(v) / len(train_loader) for k, v in iters.items()}

    plt.figure(figsize=(12, 3))
    labels = scores.keys()
    for i, label in enumerate(labels):
        plt.subplot(1, len(labels), i + 1)
        plt.plot(epochs[label], scores[label])
        plt.xlabel("Epoch")
        if i == 0:
            plt.ylabel("Score")
        plt.title(label)
    
    plt.savefig(METRICS_PATH, bbox_inches="tight")
    print(f"plot saved at {METRICS_PATH}")
    if show_plot:
        plt.show()
    plt.close()

def train_iteration(model, batch):
    model.train()

    src, tgt, flow = batch
    out = model(src.to(DEVICE), flow.to(DEVICE))
    loss = loss_fn(out, tgt.to(DEVICE))

    return loss

def evaluate(iteration, **kw):
    scores = defaultdict(float)
    for src_frame, tgt_frames, flow in tqdm(test_loader):
        assert src_frame.shape[0] == 1, src_frame.shape
        flow = flow.squeeze(0)
        tgt_frames = tgt_frames.squeeze(0)

        out_frames = iteration(src_frame.to(DEVICE), flow.to(DEVICE), **kw)
        tgt_frames = test_loader.dataset.denormalize_frame(tgt_frames).to(DEVICE)

        scores["PSNR"] += psnr(out_frames, tgt_frames).item()
        scores["SSIM"] += ssim(out_frames, tgt_frames).item()
        scores["LPIPS-" + lpip.net.net._get_name()] += lpip(out_frames, tgt_frames).item()

    scores = {k: v / len(test_loader) for k, v in scores.items()}
    return scores

def evaluate_model(model):
    model.eval()
    return evaluate(predict_tensor, model=model, transforms=test_loader.dataset)

def remap_tensor(src_frame: torch.Tensor, flow: torch.Tensor, transforms: FrameFlowProcessing, interpolation=cv2.INTER_CUBIC, return_tensor: bool = True):
    # src_frame and flow should be normalized tensors
    frame_np = transforms.deprocess_frame(src_frame)[0]
    flow = transforms.deprocess_flow(flow)
    out_frames = remap(frame_np, flow, interpolation, include_first_frame=False)
    if return_tensor:
        return torch.from_numpy(out_frames).to(src_frame.dtype).to(src_frame.device).permute(0, 3, 1, 2) / 255
    else:
        return out_frames

def test(model, test_set_idx=0):
    model.eval()
    
    if not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR)
    
    if isinstance(test_set_idx, int):
        test_set_idx = [test_set_idx]

    for idx in test_set_idx:
        seq = test_loader.dataset.data[idx]
        shot_id, start_sec, _, fps = seq

        frame, tgt_frames, flow = test_loader.dataset[idx]
        frame = frame.unsqueeze(0).to(DEVICE)
        flow = flow.to(DEVICE)

        # predictions
        pred_model = predict_tensor(frame, flow, model=model, transforms=test_loader.dataset, return_tensor=False)
        pred_model = add_video_title(pred_model, "Model prediction")
        
        pred_soft = softsplat_tensor(frame, flow, transforms=test_loader.dataset, weight_type=None, return_tensor=False)
        pred_soft = add_video_title(pred_soft, "Average Splatting")
        
        pred_remap = remap_tensor(frame, flow, transforms=test_loader.dataset, interpolation=cv2.INTER_CUBIC, return_tensor=False)
        pred_remap = add_video_title(pred_remap, "OpenCV Remapping")
        
        # results
        tgt_frames = add_video_title(test_loader.dataset.deprocess_frame(tgt_frames), "Original video")
        res = np.concatenate([
            np.concatenate([tgt_frames, pred_model], axis=2),
            np.concatenate([pred_remap, pred_soft], axis=2),
        ], axis=1)
        ts = datetime.datetime.now().isoformat().replace(":", "_")
        path = os.path.join(OUT_DIR, f"test{idx}_{shot_id}_{start_sec:03d}_{ts}.mp4")
        save_video(res, fps, path)
        print(f"results saved at {path}")

if __name__ == "__main__":
    training = Training(model, optimizer, lr_scheduler=None, ckpt_path=MODEL_PATH)
    training.run(
        max_niters=len(train_loader) * 300,
        train_loader=train_loader,
        train_iteration=train_iteration,
        evaluate=evaluate_model,
        test=test,
        plot_loss=plot_loss,
        print_step=5,
        plot_step=len(train_loader),
        save_step=len(train_loader),
        eval_step=len(train_loader) * 5,
        test_step=len(train_loader) * 5,
    )
    