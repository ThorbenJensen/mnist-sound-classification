import wget
import zipfile

# %% DOWNLOAD ZIP
zip_source = \
    'https://github.com/Jakobovski/free-spoken-digit-dataset/archive/v1.0.8.zip'

zip_target = 'data/free-spoken-digit-dataset.zip'

wget.download(url=zip_source, out=zip_target)

# %% UNZIP

with zipfile.ZipFile(zip_target,'r') as zip_ref:
    zip_ref.extractall('data/.')
