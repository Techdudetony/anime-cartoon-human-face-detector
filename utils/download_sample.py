import os
import random
import shutil # Shell Utilities
import kagglehub
import zipfile
from pathlib import Path
from tqdm import tqdm # Progress Bar in Terminal

def download_sample(dataset_id, category_name, sample_size=2000):
    # Download dataset using Kagglehub
    print(f"\n Downloading '{dataset_id} via kagglehub...")
    path = kagglehub.dataset_download(dataset_id)
    path = Path(path)
    
    # Find ZIP file 
    zip_files = list(path.glob('*.zip'))
    if zip_files:
        print("Extracting Zip file...")
        with zipfile.ZipFile(zip_files[0], 'r') as zip_ref:
            zip_ref.extractall(path)
        zip_files[0].unlink() # Delete zip to save space
    
    # Locate all image files
    image_ext = ['.jpg', '.jpeg', '.png']
    all_images = [f for f in path.rglob('*') if f.suffix.lower() in image_ext]
    print(f"Found {len(all_images)} total images in '{category_name} dataset.")
    
    # Sample and copy to category folder
    os.makedirs(f"dataset/{category_name}", exist_ok=True)
    sample_images = random.sample(all_images, min(sample_size, len(all_images)))
    
    print(f"Copying {len(sample_images)} images to dataset/{category_name}...")
    for img in tqdm(sample_images):
        dest = Path(f"dataset/{category_name}/{img.name}")
        shutil.copy(img, dest)
    
    print(f"Finished category: {category_name}\n")
    
if __name__ == "__main__":
    # Anime Faces
    download_sample("splcher/animefacedataset", "anime", sample_size=2000)
    
    # Cartoon Faces
    download_sample("brendanartley/cartoon-faces-googles-cartoon-set", "cartoon", sample_size=2000)
    
    # Human Faces
    download_sample("ashwingupta3012/human-faces", "human", sample_size=2000)