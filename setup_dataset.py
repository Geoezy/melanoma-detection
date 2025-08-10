import os
import shutil
import zipfile
from kaggle.api.kaggle_api_extended import KaggleApi

os.system('pip install kaggle pandas scikit-learn --quiet')


kaggle_secret_content = os.environ.get('KAGGLE_JSON')

if not kaggle_secret_content:
    raise ValueError("KAGGLE_JSON secret not found. Please add it to your repository settings.")

kaggle_dir = os.path.join(os.path.expanduser('~'), '.kaggle')
os.makedirs(kaggle_dir, exist_ok=True)
kaggle_json_path = os.path.join(kaggle_dir, 'kaggle.json')

with open(kaggle_json_path, 'w') as f:
    f.write(kaggle_secret_content)

os.chmod(kaggle_json_path, 0o600)

api = KaggleApi()
api.authenticate()

api.dataset_download_files('kmader/skin-cancer-mnist-ham10000', path='.', quiet=True)

zip_filename = 'skin-cancer-mnist-ham10000.zip'
with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
    zip_ref.extractall('.')