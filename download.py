import gdown
import tarfile
from tqdm import tqdm
import os

# Download the tar file
url = 'https://drive.google.com/u/1/uc?id=1CytKKfWgPZ9iRTxoducpp5ovaN6vLeUE&export=download'
output = 'data.tar.gz'

gdown.download(url, output, quiet=False)

# extract the tar file
tar = tarfile.open(output)
tar.extractall()
tar.close()

# remove the tar file

os.remove(output)
