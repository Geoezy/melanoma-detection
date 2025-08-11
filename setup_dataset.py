import os
import shutil
import zipfile
import pandas as pd
import pydicom
from PIL import Image
from sklearn.model_selection import train_test_split
from kaggle.api.kaggle_api_extended import KaggleApi

api = KaggleApi()
api.authenticate()

api.competition_download_files('isic-2024-challenge', path='.', quiet=False)

with zipfile.ZipFile('isic-2024-challenge.zip', 'r') as zip_ref:
    zip_ref.extractall('.')

if os.path.exists('train-image.zip'):
    with zipfile.ZipFile('train-image.zip', 'r') as zip_ref:
        zip_ref.extractall('train-image')

print("--- Download and Unzip Complete ---")

dicom_dir = os.path.join('train-image', 'image')
png_dir = 'isic2024_png_images'
os.makedirs(png_dir, exist_ok=True)

def convert_dicom_to_png(dicom_path, png_path):
    try:
        dicom_file = pydicom.dcmread(dicom_path)
        pixel_array = dicom_file.pixel_array
        image = Image.fromarray(pixel_array)
        image.save(png_path)
        return True
    except Exception as e:
        return False

for filename in os.listdir(dicom_dir):
    if filename.endswith('.dcm'):
        dicom_path = os.path.join(dicom_dir, filename)
        png_path = os.path.join(png_dir, os.path.splitext(filename)[0] + '.png')
        convert_dicom_to_png(dicom_path, png_path)
metadata = pd.read_csv('train-metadata.csv')

metadata['png_path'] = metadata['isic_id'].apply(
    lambda x: os.path.join(png_dir, x + '.png')
)

metadata['label'] = metadata['target'].apply(
    lambda x: 'Malignant' if x == 1 else 'Benign'
)

train_df, test_df = train_test_split(
    metadata,
    test_size=0.2,
    stratify=metadata['label'],
    random_state=42
)

def organize_images(dataframe, split_name):
    for index, row in dataframe.iterrows():
        label = row['label']
        original_path = row['png_path']
       
        destination_folder = os.path.join('isic2024_organized', split_name, label)
        os.makedirs(destination_folder, exist_ok=True)
       
        if os.path.exists(original_path):
            shutil.copy(original_path, destination_folder)

organize_images(train_df, 'train')
organize_images(test_df, 'test')
