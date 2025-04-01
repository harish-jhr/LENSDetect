import numpy as np
import os
from tqdm import tqdm  # For progress bar

def create_lens_npz(data_dir, save_path, folders):
    """Loads images from data_dir folders and saves them as a single .npz file"""
    
    labels_map = {folders[0]: 1, folders[1]: 0}  #labels (Lenses → 1, Non-Lenses → 0)
    all_images = []
    all_labels = []

    for category in folders:
        folder_path = os.path.join(data_dir, category)

        if not os.path.exists(folder_path):  # Ensure folder exists
            print(f"{folder_path} does not exist, skipping.")
            continue

        files = sorted(os.listdir(folder_path))  # Keep numerical order

        for file in tqdm(files, desc=f"Loading {category}"):
            file_path = os.path.join(folder_path, file)
            img = np.load(file_path)  # Loading image array

            if img.shape != (3, 64, 64):  # Ensure correct shape
                print(f"Skipping {file_path}, shape: {img.shape}")
                continue

            all_images.append(img)
            all_labels.append(labels_map[category])  # Assign label

    all_images = np.array(all_images, dtype=np.float32)  # Convert to NumPy array
    all_labels = np.array(all_labels, dtype=np.int64)

    np.savez_compressed(save_path, images=all_images, labels=all_labels)  # Save as .npz
    print(f"Saved dataset to {save_path}, shape: {all_images.shape}")
