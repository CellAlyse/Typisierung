import gdown
from tqdm import tqdm
import os
import zipfile

# Download the tar file
url = 'https://drive.google.com/u/1/uc?id=1Xprfr6NqBSUbI-kJau_hma_DZbEuHqXa&export=download'
output = 'data.zip'

gdown.download(url, output, quiet=False)

# extract the zip file
with zipfile.ZipFile(output, 'r') as zip_ref:
    zip_ref.extractall('.')

# remove the zip file
os.remove(output)