# Audio-Visual Separation with Hierarchical Fusion and Representation Alignment

- [Introduction](#introduction)

- [Citing](#citing)

- [Data Preparation](#data-preparation)

- [Run the Code](#run-the-code)

- [Checkpoints](#checkpoints)

- [Contact](#contact)

## Introduction
This work proposes a hierarchical fusion strategy that combines middle and late fusion to handle diverse acoustic properties, and a representation alignment approach that aligns U-Net audio features with CLAP embeddings to enrich semantic information.Our method achieves state-of-the-art SDR results on MUSIC, MUSIC-21, and VGGSound datasets under the self-supervised setting.


## Citing
Please cite our paper if you find this repository useful.
```

@inproceedings{,

title={Audio-Visual Separation with Hierarchical Fusion and Representation Alignment},

author={Han Hu and Dongheng Lin  and Qiming Huang  and Yuqi Hou  and Hyung Jin Chang  and Jianbo Jiao},

booktitle={The Thirty Sixth British Machine Vision Conference},

year={2025},

url={}

}
```

## Data Preparation
#### Step 1. Extract audio track and image frames from the video


We organize the extracted data as follows:
```text
dataset/
├── audio/
│   ├── {instrument}/
│   │   ├── {video_id_1}.wav
│   │   ├── {video_id_2}.wav
│   │   └── ...
│   └── ...
└── frames/
    ├── {instrument}/
    │   ├── {video_id_1}/
    │   │   ├── 000001.jpg
    │   │   ├── 000002.jpg
    │   │   └── ...
    │   ├── {video_id_2}/
    │   │   ├── 000001.jpg
    │   │   └── ...
    │   └── ...
    └── ...
```

* **Audio:** resample every track to 11,625 Hz and save as `{video_id}.wav` under its instrument class.
* **Frames:** extract frames at each video’s native frame rate, saving sequential images inside `{video_id}/`.

You can see how we automatically extract frames from raw videos in `utils_func/frame_extraction.py`.


#### Step 2. Build training/validation CSV files

We provide example CSVs for the MUSIC dataset under the `metadata/` folder.
Each row records the audio path, the frames directory, and the number of frames in that clip. `n_frames` is simply the count of `*.jpg` files inside the corresponding `{video_id}/` folder.

**Example (`metadata/train.csv`):**

```csv
dataset/audio/violin/vid_0001.wav,dataset/frames/violin/000001,156
dataset/audio/cello/vid_0101.wav,dataset/frames/cello/000101,298
```


#### Step 3. Precompute CLIP visual features.


We provide an automatic precompute script that  encodes extracted frames with CLIP, and saves the results into a single HDF5 file. 

* Script: `utils_func/MUSIC21_video_precompute.py`
* Backbone: CLIP ViT-B/32 (configurable)
* Sampling: 1 feature every 8 frames (configurable)

Precomputing features significantly reduces training time and memory overhead.



## Run the Code
#### 1) Build a virtual environment and install packages

```bash
conda create -n av-sep python=3.10 -y
conda activate av-sep

# Install PyTorch (pick the command for your CUDA/OS from https://pytorch.org/get-started/locally/)
# Example (CUDA 12.1):
pip install --index-url https://download.pytorch.org/whl/cu121 torch torchvision

# Core deps
pip install numpy scipy tqdm librosa mir_eval h5py pandas opencv-python

# OpenAI CLIP (official)
pip install git+https://github.com/openai/CLIP.git#egg=clip
```

**LAION-CLAP**

* Please follow the official instructions here: [https://github.com/LAION-AI/CLAP](https://github.com/LAION-AI/CLAP)
  (CLAP may download weights on first use.)


#### 2) Prepare inputs
Make sure you have:

* Correct paths for `train.csv` and `val.csv` (e.g., under `metadata/`).
* A precomputed CLIP HDF5 file and its `h5_file_path`.
* Properly set `out_dir`, `train_list`, `val_list` in your script or via CLI flags.

### 3) (Optional) Toggle representation alignment

We expose alignment via the dataset flag `audio_embeding` (default `True`).
To disable alignment, set it to `False` when constructing both train/val datasets:

```python
train_dataset = dataset.MixDataset_general(..., audio_embeding=False)
val_dataset   = dataset.MixDataset_general(..., audio_embeding=False)
```


## Checkpoints

You can download our pretrained checkpoint here:

- [model_131.pt](https://drive.google.com/file/d/1MiyCQemM52eoZTvWziFlJpwopHKnWw7R/view?usp=share_link)


## Contact
If you have a question, please bring up an issue (preferred) or send me an email [hxh347@student.bham.ac.uk](hxh347@student.bham.ac.uk).
