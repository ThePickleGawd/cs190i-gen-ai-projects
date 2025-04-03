import os
import tarfile
import urllib.request

url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
dirname = os.path.dirname(__file__)
filepath = os.path.join(dirname, "imagenette2-320.tgz")

# Download
if not os.path.exists(filepath):
    urllib.request.urlretrieve(url, filepath)

# Extract
with tarfile.open(filepath, "r:gz") as tar:
    tar.extractall(path=dirname)
