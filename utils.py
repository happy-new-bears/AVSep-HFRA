"""Utility functions."""
import contextlib
import csv
import json
import os
import pathlib
import subprocess as sp
import warnings
from threading import Timer

#import cv2
import librosa
import numpy as np

import torch
import collections
import torch.nn as nn
import torch.nn.functional as F


import mir_eval.separation
import mir_eval

def save_args(filename, args):
    """Save the command-line arguments."""
    args_dict = {}
    for key, value in vars(args).items():
        if isinstance(value, pathlib.Path):
            args_dict[key] = str(value)
        else:
            args_dict[key] = value
    save_json(filename, args_dict)


def inverse_dict(d):
    """Return the inverse dictionary."""
    return {v: k for k, v in d.items()}


def save_txt(filename, data):
    """Save a list to a TXT file."""
    with open(filename, "w", encoding="utf8") as f:
        for item in data:
            f.write(f"{item}\n")


def load_txt(filename):
    """Load a TXT file as a list."""
    with open(filename, encoding="utf8") as f:
        return [line.strip() for line in f]


def save_json(filename, data):
    """Save data as a JSON file."""
    with open(filename, "w", encoding="utf8") as f:
        json.dump(data, f)


def load_json(filename):
    """Load data from a JSON file."""
    with open(filename, encoding="utf8") as f:
        return json.load(f)


def save_csv(filename, data, fmt="%d", header=""):
    """Save data as a CSV file."""
    np.savetxt(
        filename, data, fmt=fmt, delimiter=",", header=header, comments=""
    )


def load_csv(filename, skiprows=1):
    """Load data from a CSV file."""
    return np.loadtxt(filename, dtype=int, delimiter=",", skiprows=skiprows)


def load_csv_text(filename, headerless=True):
    """Read a CSV file into a list of dictionaries or lists."""
    with open(filename) as f:
        if headerless:
            return [row for row in csv.reader(f)]
        reader = csv.DictReader(f)
        return [
            {field: row[field] for field in reader.fieldnames}
            for row in reader
        ]


def ignore_exceptions(func):
    """Decorator that ignores all errors and warnings."""

    def inner(*args, **kwargs):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                return func(*args, **kwargs)
            except Exception:
                return None

    return inner


def suppress_outputs(func):
    """Decorator that suppresses writing to stdout and stderr."""

    def inner(*args, **kwargs):
        devnull = open(os.devnull, "w")
        with contextlib.redirect_stdout(devnull):
            with contextlib.redirect_stderr(devnull):
                return func(*args, **kwargs)

    return inner


def resolve_paths(func):
    """Decorator that resolves all paths."""

    def inner(*args, **kwargs):
        parsed = func(*args, **kwargs)
        for key in vars(parsed).keys():
            if isinstance(getattr(parsed, key), pathlib.Path):
                setattr(
                    parsed, key, getattr(parsed, key).expanduser().resolve()
                )
        return parsed

    return inner


def warpgrid(bs, HO, WO, warp=True):
    # meshgrid
    x = np.linspace(-1, 1, WO)
    y = np.linspace(-1, 1, HO)
    xv, yv = np.meshgrid(x, y)
    grid = np.zeros((bs, HO, WO, 2))
    grid_x = xv
    if warp:
        grid_y = (np.power(21, (yv + 1) / 2) - 11) / 10
    else:
        grid_y = np.log(yv * 10 + 11) / np.log(21) * 2 - 1
    grid[:, :, :, 0] = grid_x
    grid[:, :, :, 1] = grid_y
    grid = grid.astype(np.float32)
    return grid


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        val = np.asarray(val)
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        if self.val is None:
            return 0.0
        else:
            return self.val.tolist()

    def average(self):
        if self.avg is None:
            return 0.0
        else:
            return self.avg.tolist()


def recover_rgb(img):
    for t, m, s in zip(img, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]):
        t.mul_(s).add_(m)
    img = (img.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    return img


def recover_rgb_clip(img):
    for t, m, s in zip(
        img,
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711],
    ):
        t.mul_(s).add_(m)
    img = (img.numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
    return img


def magnitude2heatmap(mag, log=True, scale=200.0):
    if log:
        mag = np.log10(mag + 1.0)
    mag *= scale
    mag[mag > 255] = 255
    mag = mag.astype(np.uint8)
    # mag_color = cv2.applyColorMap(mag, cv2.COLORMAP_JET)
    mag_color = cv2.applyColorMap(mag, cv2.COLORMAP_INFERNO)
    mag_color = mag_color[:, :, ::-1]
    return mag_color


def istft_reconstruction(mag, phase, hop_len, win_len):
    #spec = mag.astype(np.complex) * np.exp(1j * phase)
    spec = mag.astype(complex) * np.exp(1j * phase)
    wav = librosa.istft(spec, hop_length=hop_len, win_length=win_len)
    return np.clip(wav, -1.0, 1.0).astype(np.float32)


class VideoWriter:
    """ Combine numpy frames into video using ffmpeg

    Arguments:
        filename: name of the output video
        fps: frame per second
        shape: shape of video frame

    Properties:
        add_frame(frame):
            add a frame to the video
        add_frames(frames):
            add multiple frames to the video
        release():
            release writing pipe

    """

    def __init__(self, filename, fps, shape):
        self.file = filename
        self.fps = fps
        self.shape = shape

        # video codec
        ext = filename.split(".")[-1]
        if ext == "mp4":
            self.vcodec = "h264"
        else:
            raise RuntimeError("Video codec not supoorted.")

        # video writing pipe
        cmd = [
            "ffmpeg",
            "-y",  # overwrite existing file
            "-f",
            "rawvideo",  # file format
            "-s",
            "{}x{}".format(shape[1], shape[0]),  # size of one frame
            "-pix_fmt",
            "rgb24",  # 3 channels
            "-r",
            str(self.fps),  # frames per second
            "-i",
            "-",  # input comes from a pipe
            "-an",  # not to expect any audio
            "-vcodec",
            self.vcodec,  # video codec
            "-pix_fmt",
            "yuv420p",  # output video in yuv420p
            self.file,
        ]

        self.pipe = sp.Popen(
            cmd, stdin=sp.PIPE, stderr=sp.PIPE, bufsize=10 ** 9
        )

    def release(self):
        self.pipe.stdin.close()

    def add_frame(self, frame):
        assert len(frame.shape) == 3
        assert frame.shape[0] == self.shape[0]
        assert frame.shape[1] == self.shape[1]
        try:
            self.pipe.stdin.write(frame.tostring())
        except:
            _, ffmpeg_error = self.pipe.communicate()
            print(ffmpeg_error)

    def add_frames(self, frames):
        for frame in frames:
            self.add_frame(frame)


def kill_proc(proc):
    proc.kill()
    print("Process running overtime! Killed.")


def run_proc_timeout(proc, timeout_sec):
    # kill_proc = lambda p: p.kill()
    timer = Timer(timeout_sec, kill_proc, [proc])
    try:
        timer.start()
        proc.communicate()
    finally:
        timer.cancel()


def combine_video_audio(src_video, src_audio, dst_video, verbose=False):
    try:
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "quiet",
            "-i",
            src_video,
            "-i",
            src_audio,
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-strict",
            "experimental",
            dst_video,
        ]
        proc = sp.Popen(cmd)
        run_proc_timeout(proc, 10.0)

        if verbose:
            print("Processed:{}".format(dst_video))
    except Exception as e:
        print("Error:[{}] {}".format(dst_video, e))


# save video to the disk using ffmpeg
def save_video(path, tensor, fps=25):
    assert tensor.ndim == 4, "video should be in 4D numpy array"
    L, H, W, C = tensor.shape
    writer = VideoWriter(path, fps=fps, shape=[H, W])
    for t in range(L):
        writer.add_frame(tensor[t])
    writer.release()


def save_audio(path, audio_numpy, sr):
    librosa.output.write_wav(path, audio_numpy, sr)

    
## add more function by Han
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val*weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        val = np.asarray(val)
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        if self.val is None:
            return 0.
        else:
            return self.val.tolist()

    def average(self):
        if self.avg is None:
            return 0.
        else:
            return self.avg.tolist()

def calc_metrics(batch_data, outputs, args):
    # meters
    sdr_mix_meter = AverageMeter()
    sdr_meter = AverageMeter()
    sir_meter = AverageMeter()
    sar_meter = AverageMeter()

    # fetch data and predictions
    mag_mix = batch_data['mag_mix']
    phase_mix = batch_data['phase_mix']
    audios = batch_data['audios']

    pred_masks_ = outputs['pred_masks']

    # unwarp log scale
    N = 2
    B = mag_mix.size(0)
    pred_masks_linear = [None for n in range(N)]
    for n in range(N):
        if args.log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(B, args.stft_frame//2+1, pred_masks_[0].size(3), warp=False)).to(args.device)
            pred_masks_linear[n] = F.grid_sample(pred_masks_[n], grid_unwarp)
        else:
            pred_masks_linear[n] = pred_masks_[n]

    # convert into numpy
    mag_mix = mag_mix.numpy()
    phase_mix = phase_mix.numpy()
    for n in range(N):
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()

        # threshold if binary mask
        if args.binary_mask:
            pred_masks_linear[n] = (pred_masks_linear[n] > args.mask_thres).astype(np.float32)

    # loop over each sample
    for j in range(B):
        # save mixture
        mix_wav = istft_reconstruction(mag_mix[j, 0], phase_mix[j, 0], hop_length=args.stft_hop)

        # save each component
        preds_wav = [None for n in range(N)]
        for n in range(N):
            # Predicted audio recovery
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            preds_wav[n] = istft_reconstruction(pred_mag, phase_mix[j, 0], hop_length=args.stft_hop)

        # separation performance computes
        L = preds_wav[0].shape[0]
        gts_wav = [None for n in range(N)]
        valid = True
        for n in range(N):
            gts_wav[n] = audios[n][j, 0:L].numpy()
            valid *= np.sum(np.abs(gts_wav[n])) > 1e-5
            valid *= np.sum(np.abs(preds_wav[n])) > 1e-5
        if valid:
            sdr, sir, sar, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray(preds_wav),
                False)
            sdr_mix, _, _, _ = bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray([mix_wav[0:L] for n in range(N)]),
                False)
            sdr_mix_meter.update(sdr_mix.mean())
            sdr_meter.update(sdr.mean())
            sir_meter.update(sir.mean())
            sar_meter.update(sar.mean())

    return [sdr_mix_meter.average(),
            sdr_meter.average(),
            sir_meter.average(),
            sar_meter.average()]


def calc_metrics(
    batch,
    out,
    n_mix=2,
    n_fft=1024,
    hop_len=256,
    win_len=1024,
    use_log_freq=True,
    use_binary_mask=True,
    #backend="mir_", # mus_eval
    threshold=0.5,
    image_model="clip",
    include_pit=True,
):
    # Initialize lists
    metrics = collections.defaultdict(list)

    # Fetch data and predictions
    mag_mix = batch["mag_mix"]
    phase_mix = batch["phase_mix"]
    audios = batch["audios"]
    
    #pred_masks = [out["pred_masks"][0], 1 - out["pred_masks"][0]] # （原始掩码，原始掩码的反转）
    pred_masks = [out["pred_masks"][0], out["pred_masks"][1]]

    # Unwarp log scale
    N = n_mix
    B = mag_mix.size(0)
    #print('check the batch:',B)
    pred_masks_linear = [None] * N
    for n in range(N):
        if use_log_freq:
            grid_unwarp = torch.from_numpy(
                warpgrid(
                    B,
                    n_fft // 2 + 1,
                    pred_masks[0].size(3),
                    warp=False,
                )
            ).to(pred_masks[n].device)
            pred_masks_linear[n] = F.grid_sample(
                pred_masks[n], grid_unwarp, align_corners=True
            )
        else:
            pred_masks_linear[n] = pred_masks[n]


    # Convert into numpy arrays
    mag_mix = mag_mix.detach().cpu().numpy()
    phase_mix = phase_mix.detach().cpu().numpy()
    for n in range(N):
        pred_masks_linear[n] = pred_masks_linear[n].detach().cpu().numpy()

        # Apply a threshold
        if use_binary_mask:
            pred_masks_linear[n] = (pred_masks_linear[n] > threshold).astype(
                np.float32
            )
    # Loop over each sample
    
    for j in range(B):
        #print('check mag_mix:',sum(mag_mix[j, 0]))
        # Reconstruct the mixture
        mix_wav = istft_reconstruction(
            mag_mix[j, 0], phase_mix[j, 0], hop_len=hop_len, win_len=win_len
        )
        #print('check the mix_wav',mix_wav.shape)
        # Reconstruct each component
        pred_wavs = [None] * N
        for n in range(N):
            pred_mag = mag_mix[j, 0] * pred_masks_linear[n][j, 0]
            pred_wavs[n] = istft_reconstruction(
                pred_mag, phase_mix[j, 0], hop_len=hop_len, win_len=win_len
            )
            #print('check the pred_wavs',pred_wavs[n].shape)
        # Compute separation performance
        L = pred_wavs[0].shape[0]
        #print('L = {}'.format(L))
        gts_wav = [None] * N
        valid = True
        for n in range(N):
            gts_wav[n] = audios[n][j, 0:L].numpy()
            valid *= np.sum(np.abs(gts_wav[n])) > 1e-5
            valid *= np.sum(np.abs(pred_wavs[n])) > 1e-5
        if not valid: #！！！！！！！！this part cannot be deleted during training
            print('check if valid:',np.abs(pred_wavs[n]))
            continue
        # print(len(pred_wavs), pred_wavs[0].shape)
        #if backend == "mir_eval": 
        if True:
            sdr, sir, sar, _ = mir_eval.separation.bss_eval_sources(
                np.asarray(gts_wav), np.asarray(pred_wavs), False
            )
            (
                sdr_mix,
                sir_mix,
                sar_mix,
                _,
            ) = mir_eval.separation.bss_eval_sources(
                np.asarray(gts_wav),
                np.asarray([mix_wav[0:L]] * N),
                False,
            )
        else:
            raise ValueError(f"Unknown backend : {backend}.")
        
        metrics["sdr"].extend(sdr.tolist())
        metrics["sir"].extend(sir.tolist())
        metrics["sar"].extend(sar.tolist())

        metrics["sdr_mix"].extend(sdr_mix.tolist())
        metrics["sir_mix"].extend(sir_mix.tolist())
        metrics["sar_mix"].extend(sar_mix.tolist())
    return metrics


@torch.no_grad()
def sinkhorn(Q: torch.Tensor, nmb_iters: int, world_size=1) -> torch.Tensor:
    with torch.no_grad():
        Q = Q.detach().clone()
        sum_Q = torch.sum(Q)
        if world_size > 1:
            dist.all_reduce(sum_Q)
        Q /= sum_Q
        K, B = Q.shape
        u = torch.zeros(K).to(Q.device)
        r = torch.ones(K).to(Q.device) / K
        c = torch.ones(B).to(Q.device) / B * world_size

        if world_size > 1:
            curr_sum = torch.sum(Q, dim=1)
            dist.all_reduce(curr_sum)

        for _ in range(nmb_iters):
            if world_size > 1:
                u = curr_sum
            else:
                u = torch.sum(Q, dim=1)
            Q *= (r / u).unsqueeze(1)
            Q *= (c / torch.sum(Q, dim=0)).unsqueeze(0)
            if world_size > 1:
                curr_sum = torch.sum(Q, dim=1)
                dist.all_reduce(curr_sum)

        return (Q / torch.sum(Q, dim=0, keepdim=True)).t().float()