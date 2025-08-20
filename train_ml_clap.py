import argparse
import logging
import pathlib
import pprint
import random
import shutil
import sys
import types

import clip
import numpy as np
import torch
import torch.optim
import torch.utils.data
import torchvision.models
import tqdm

#import clipsep
import dataset
import utils

## for model 
#from clipsep import UNet
import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
import librosa
import pickle

import mir_eval.separation
import scipy

import collections
import laion_clap

from clipsep_ml_clap import CLIPSep_ML

from utils import calc_metrics




#---------------------------step 1. Add all arguments-----------
class Args:
    def __init__(self):
        # File paths and directories
        self.out_dir = pathlib.Path("/bask/projects/j/jiaoj-rep-learn/huhan/audio_visual_seperation/logs/clip_ml_clap")
        self.train_list = pathlib.Path("/bask/projects/j/jiaoj-rep-learn/Dataset/MUSIC/11solo_9duet/train.csv")
        self.val_list = pathlib.Path("/bask/projects/j/jiaoj-rep-learn/Dataset/MUSIC/11solo_9duet/val.csv")
        self.n_validation =  120 #  number of samples to evaluate # here double check
        #self.weights = pathlib.Path("/path/to/pretrained_weights.pth") # do not have pretrained weights now

        # Data parameters
        self.batch_size = 32
        self.drop_closest = None
        self.drop_closest_steps = 10000
        self.repeat = 100 # here double check
        self.frame_margin = 10  # the number of starting and ending frames to exclude
        self.audio_only = False

        # Audio parameters
        self.audio_len = 65535
        self.audio_rate = 11625 # double check needed
        self.n_fft = 1024
        self.hop_len = 256
        self.win_len = 1024

        # Image parameters
        self.img_size = 224
        self.fps = 8 # video frame sampling rate #  it is 8 in my case

        # Model parameters
        self.train_mode = "image"
        self.image_model = "clip"
        self.n_mix = 2
        self.fusion = "late"
        self.channels = 32
        self.layers = 7
        self.frames = 3
        self.stride_frames = 1 # sampling stride of frames # double check needed
        self.binary_mask = True
        self.loss = "bce"
        self.weighted_loss = True
        self.log_freq = True

        # Training parameters
        self.steps = 200000
        self.valid_steps = 10000 # 
        self.lr = 0.001 
        self.lr_warmup_steps = 5000
        self.lr_decay_steps = 100000
        self.lr_decay_multiplier = 0.1
        self.grad_norm_clip = 1.0


        # Other parameters

        self.seed = 1234
        
        '''
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"Number of GPUs available: {gpu_count}")
        else:
            gpu_count = 0
            print("CUDA is not available. Running on CPU.")

        self.gpus = gpu_count
        '''
        
        self.gpus = 1
        self.workers = 24
        self.quiet = False # show warnings only

# Instantiate the Args class to create an object with the hardcoded parameters
args = Args()


def count_parameters(net):
    """Return the number of parameters in a model."""
    return sum(p.numel() for p in net.parameters())


def get_lr_multiplier(
    step, warmup_steps, decay_end_steps, decay_end_multiplier
):
    """Return the learning rate multiplier with a warmup and decay schedule.
    The learning rate multiplier starts from 0 and linearly increases to 1
    after `warmup_steps`. After that, it linearly decreases to
    `decay_end_multiplier` until `decay_end_steps` is reached.
    """
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step > decay_end_steps:
        return decay_end_multiplier
    position = (step - warmup_steps) / (decay_end_steps - warmup_steps)
    return 1 - (1 - decay_end_multiplier) * position

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

# Make sure the checkpoint and sample directories exist
(args.out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
(args.out_dir / "samples").mkdir(exist_ok=True)
(args.out_dir / "samples" / "text").mkdir(exist_ok=True)
(args.out_dir / "samples" / "image").mkdir(exist_ok=True)

device = torch.device("cuda")

import wandb
wandb.init(project="train_audio_rep", config=args.__dict__)

#---------------------
logging.info(f"Creating the model...")
model = CLIPSep_ML(
            args.n_mix,
            args.layers,
            args.channels,
            use_log_freq=args.log_freq,
            use_weighted_loss=args.weighted_loss,
            use_binary_mask=args.binary_mask,)

model = torch.nn.DataParallel(model, device_ids=range(args.gpus))
model.to(device)
logging.info(f"------------------Total number of parameters:------------ {count_parameters(model)}")
print(f"------------------Total number of parameters:------------ {count_parameters(model)}")


#-------------------------------data loading-----------------------
# Datasets and loaders
logging.info("Creating the data loaders...")

train_dataset = dataset.MixDataset_general(
        args.train_list,
        "train",
        n_mix=args.n_mix,
        audio_len=args.audio_len,
        audio_rate=args.audio_rate,
        n_fft=args.n_fft,
        hop_len=args.hop_len,
        win_len=args.win_len,
        n_frames=args.frames,
        stride_frames=args.stride_frames,
        img_size=args.img_size,
        fps=args.fps,
        preprocess_func=dataset.transform(),
        return_waveform=False,
        repeat=args.repeat,
        frame_margin=args.frame_margin,
        precompute = True,
        h5_file_path='/bask/projects/j/jiaoj-rep-learn/Dataset/MUSIC/11solo_9duet/precomputed_VITB32_8frame_features.h5',
        audio_embeding = True
    )
len(train_dataset)

if args.repeat is None:
    logging.info(f"Training set size: {len(train_dataset)}")
else:
    logging.info(f"Training set size: {len(train_dataset) // args.repeat}")
    
train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
    )
val_dataset = dataset.MixDataset_general(
        args.val_list,
        "valid",
        n_mix=args.n_mix,
        audio_len=args.audio_len,
        audio_rate=args.audio_rate,
        n_fft=args.n_fft,
        hop_len=args.hop_len,
        win_len=args.win_len,
        n_frames=args.frames,
        stride_frames=args.stride_frames,
        img_size=args.img_size,
        fps=args.fps,
        preprocess_func=dataset.transform(),
        return_waveform=False,
        precompute = True,
        h5_file_path='/bask/projects/j/jiaoj-rep-learn/Dataset/MUSIC/11solo_9duet/precomputed_VITB32_8frame_features.h5',
        audio_embeding = True
    )

logging.info(f"Validation set size: {len(val_dataset)}")
val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False, 
    )


optimizer = torch.optim.Adam(model.parameters(), args.lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr_multiplier(
            step,
            args.lr_warmup_steps,
            args.lr_decay_steps,
            args.lr_decay_multiplier,
        ),
    )

# Create a file to record losses
loss_history = []
if args.audio_only or "clip" in args.image_model:
        loss_header = "step,train_loss,sdr, sar, sir"
else:
        loss_header = "step,train_loss,val_loss_text,val_loss_img"


# Initialize variables
step = 0
min_val_loss = float("inf")

# --------------------begin to train----------------

def train_one_epoch(model, train_loader, optimizer, scheduler, args, epoch):
    """
    Function to train the model for one epoch.
    """
    logging.info(f"Starting epoch {epoch + 1}/{args.required_epochs}...")
    model.train()
    pbar = tqdm.tqdm(train_loader, ncols=120)
    
    #count = 0 # !!!! just for debug
    for batch in pbar:
        optimizer.zero_grad()
        loss, out = model.forward(batch)
        loss = loss.mean()
        loss.backward()
        if args.grad_norm_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clip)

        optimizer.step()
        scheduler.step()
        train_loss = loss
        pbar.set_postfix(loss=f"{train_loss:.4f}")

        wandb.log({
            "train_loss": train_loss.item(),
            "epoch": epoch
        })
    return train_loss



def validate(model, val_loader,  args):
    """
    Function to validate the model.
    """
    logging.info("Validating...")
    model.eval()
    val_losses = {}
    val_modes = ["image"]

    for mode in val_modes:
        with torch.no_grad():
            total_loss = 0
            count = 0
            metrics = collections.defaultdict(list)  # Initialize metrics
            pbar = tqdm.tqdm(val_loader, ncols=120)
            
            for i, batch in enumerate(pbar):
                loss, out = model.forward(batch)
                loss = loss.mean()
                pbar.set_postfix(loss=f"{loss:.4f}")

                B = batch["mag_mix"].size(0)
                total_loss += B * float(loss)
                count += B

                batch_metrics = calc_metrics(
                    batch,
                    out,
                    n_mix=args.n_mix,
                    n_fft=args.n_fft,
                    hop_len=args.hop_len,
                    win_len=args.win_len,
                    use_log_freq=args.log_freq,
                    use_binary_mask=args.binary_mask,
                    #backend=args.backend,
                    #threshold=args.threshold,
                    image_model=args.image_model,
                    #include_pit=args.pit,
                )

                #print('----------here is the batch_metrics:',batch_metrics)
                # Aggregate metrics for logging
                for key in batch_metrics:
                    metrics[key].extend(batch_metrics[key])

                # Update progress bar with current batch metrics
                pbar.set_postfix(
                    loss=f"{loss:.4f}",
                    sdr=f"{np.mean(batch_metrics['sdr']):.2f}",
                    sir=f"{np.mean(batch_metrics['sir']):.2f}",
                    sar=f"{np.mean(batch_metrics['sar']):.2f}",
                )

            val_loss = total_loss / count
            val_losses[mode] = val_loss

            # Calculate overall metrics statistics for logging
            means = {key: np.mean(metrics[key]) for key in metrics}
            errs = {key: scipy.stats.sem(metrics[key]) for key in metrics}
            medians = {key: np.median(metrics[key]) for key in metrics}
            logging.info(
                f"Validation loss ({mode} query): {val_loss:.4f}\n"
                f"Evaluation results ({mode} query): \n"
                f"sdr={means['sdr']:.4f}±{errs['sdr']:.4f}, "
                f"sir={means['sir']:.4f}±{errs['sir']:.4f}, "
                f"sar={means['sar']:.4f}±{errs['sar']:.4f}\n"
                f"sdr_median={medians['sdr']:.4f}, "
                f"sir_median={medians['sir']:.4f}, "
                f"sar_median={medians['sar']:.4f}\n"
                f"sdr_mix={means['sdr_mix']:.4f}±{errs['sdr_mix']:.4f}, "
                f"sir_mix={means['sir_mix']:.4f}±{errs['sir_mix']:.4f}, "
                f"sar_mix={means['sar_mix']:.4f}±{errs['sar_mix']:.4f}\n"
                f"sdr_mix_median={medians['sdr_mix']:.4f}, "
                f"sir_mix_median={medians['sir_mix']:.4f}, "
                f"sar_mix_median={medians['sar_mix']:.4f}"
            )
            wandb.log({
                "val_loss": val_loss,
                "sdr": means['sdr'],
                "sir": means['sir'],
                "sar": means['sar'],
                "epoch": epoch
            })
    return val_losses,means['sdr'],means['sir'],means['sar']

def save_info(model, optimizer, scheduler, args, epoch, val_losses, min_val_loss):
    """
    Function to save model, optimizer, and scheduler states and update the best model.
    """
    checkpoint_filename = args.out_dir / "checkpoints" / f"model_{epoch}.pt"
    #if epoch % 5 == 0:
    if True:
        checkpoint_filename = args.out_dir / "checkpoints" / f"model_{epoch}.pt"
        torch.save(model.state_dict(), checkpoint_filename)
        logging.info(f"Saved the model to: {checkpoint_filename}")

        optimizer_filename = args.out_dir / "checkpoints" / f"optimizer_{epoch}.pt"
        torch.save(optimizer.state_dict(), optimizer_filename)
        logging.info(f"Saved the optimizer state to: {optimizer_filename}")

        scheduler_filename = args.out_dir / "checkpoints" / f"scheduler_{epoch}.pt"
        torch.save(scheduler.state_dict(), scheduler_filename)
        logging.info(f"Saved the scheduler state to: {scheduler_filename}")

    val_mode = args.train_mode
    #if val_losses[val_mode] < min_val_loss:

    if True:
        min_val_loss = val_losses[val_mode]
        # 直接保存模型到 "best_model.pt"，替代 shutil.copyfile
        best_model_filename = args.out_dir / "checkpoints" / "best_model.pt"
        torch.save(model.state_dict(), best_model_filename)
        logging.info(f"Minimum validation loss achieved: {min_val_loss:.4f}, saved the best model to: {best_model_filename}")

    return min_val_loss


    
# Calculate the number of epochs needed to achieve the desired number of steps
steps_per_epoch = len(train_loader)  # Number of batches per epoch
required_epochs = (args.steps + steps_per_epoch - 1) // steps_per_epoch  # Rounds up to ensure full coverage
args.required_epochs = required_epochs
logging.info(f"Minimum validation loss achieved: {args.required_epochs:.1f}")
# Main training loop
for epoch in range(args.required_epochs):
    train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, args, epoch)
    val_losses,sdr,sir,sar = validate(model, val_loader,  args)
    min_val_loss = save_info(model, optimizer, scheduler, args, epoch, val_losses, min_val_loss)
    loss_history.append((epoch, val_losses["image"],sdr,sir,sar))
    utils.save_csv(args.out_dir / "loss.csv", loss_history, fmt="%f", header=loss_header)
wandb.finish()