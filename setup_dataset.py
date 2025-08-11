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