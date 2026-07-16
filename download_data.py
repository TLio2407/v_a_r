import os
import zipfile
import gdown
from pathlib import Path

# 1. Configuration
# Replace this with your NEW personal file ID from your Google Drive copy
NEW_FILE_ID = '1b9F4B1tDVX8bIX4fZxsP9bduRynDUN_a'  
URL = f'https://drive.google.com/uc?id={NEW_FILE_ID}'

# Define paths
ZIP_PATH = Path('dataset.zip')
EXTRACT_DIR = Path('./data')

# 2. Download Phase
if not ZIP_PATH.exists():
    print(f"Starting download to {ZIP_PATH}...")
    try:
        # Removed 'remaining_ok=True' to fix the TypeError
        gdown.download(URL, str(ZIP_PATH), quiet=False)
        print("Download finished successfully.")
    except Exception as e:
        print(f"Download failed. Check your File ID and permissions.\nError: {e}")
        exit(1)
else:
    print(f"{ZIP_PATH} already exists. Skipping download.")

# 3. Extraction Phase
print(f"Extracting dataset to {EXTRACT_DIR}...")
EXTRACT_DIR.mkdir(parents=True, exist_ok=True)

try:
    with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
        file_list = zip_ref.namelist()
        total_files = len(file_list)
        print(f"Unzipping {total_files} files...")
        
        zip_ref.extractall(EXTRACT_DIR)
        print(f"Extraction complete! Files are ready at: {EXTRACT_DIR.resolve()}")

except zipfile.BadZipFile:
    print("Error: The downloaded file is corrupted or not a valid zip archive.")
except Exception as e:
    print(f"An error occurred during extraction: {e}")