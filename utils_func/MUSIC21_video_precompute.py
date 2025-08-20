import os
import torch
from PIL import Image
import h5py
import numpy as np
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel

# Set device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load the CLIP model and processor from the Transformers library
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)  
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32") 

def extract_clip_features_for_frames(frame_dir, interval, fps=30):
    frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith('.jpg')])
    features = []
    
    # Calculate the number of frames between each sample
    frames_per_interval = int(fps * interval)
    
    # Iterate through the frames with the interval
    for i in range(0, len(frame_files), frames_per_interval):
        frame_path = os.path.join(frame_dir, frame_files[i])
        
        # Load and preprocess the frame
        frame = Image.open(frame_path)
        inputs = processor(images=frame, return_tensors="pt").to(device)

        # Compute CLIP features using the Transformers CLIP model
        with torch.no_grad():
            feature = model.get_image_features(**inputs).cpu().numpy()
            features.append(feature)
    
    return np.vstack(features)  # Return as a single array

def save_clip_features_to_h5(root_dir, output_h5, interval=2, fps=30):
    print("start processing")
    with h5py.File(output_h5, 'w') as h5_file:
        # Traverse each sub-directory (each representing a video)
        for category in tqdm(os.listdir(root_dir), desc= "category"):
            category_path = os.path.join(root_dir, category)
            if not os.path.isdir(category_path):
                continue  # Skip non-directory files

            for video_name in tqdm(os.listdir(category_path)):
                video_path = os.path.join(category_path, video_name)
                if not os.path.isdir(video_path):
                    continue  # Skip non-directory files
                try:
                    # Extract features for frames in the video folder
                    features = extract_clip_features_for_frames(video_path, interval, fps)
                    
                    # Save the features to the HDF5 file
                    if len(features) > 0:
                        h5_file.create_dataset(f"{category}/{video_name}", data=features, compression="gzip")
                except Exception as e:
                    print(f"Failed to process {video_name} in {category}: {e}")

# Example usage
# Root directory containing all video directories (e.g., accordion, cello, etc.)
root_directory = '/bask/projects/j/jiaoj-rep-learn/Dataset/MUSIC/11solo_9duet/frames'

# Output HDF5 file
output_h5_file = '/bask/projects/j/jiaoj-rep-learn/Dataset/MUSIC/11solo_9duet/precomputed_ViT_b32_8frame_features.h5'

# Run the feature extraction and saving process
save_clip_features_to_h5(root_directory, output_h5_file, interval=1, fps=8)
