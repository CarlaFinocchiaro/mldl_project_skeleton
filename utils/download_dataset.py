import os
import shutil
import urllib.request
import zipfile

from torchvision.datasets import ImageFolder
import torchvision.transforms as T

import torch

# Scarica il dataset
url = "http://cs231n.stanford.edu/tiny-imagenet-200.zip"
zip_path = "tiny-imagenet-200.zip"
data_dir = "tiny-imagenet"

print("Downloading dataset...")
urllib.request.urlretrieve(url, zip_path)

# Estrai il dataset
print("Extracting dataset...")
with zipfile.ZipFile(zip_path, "r") as zip_ref:
    zip_ref.extractall(data_dir)

# Organizza le immagini della validation set nelle rispettive cartelle
val_dir = os.path.join(data_dir, "tiny-imagenet-200", "val")
annotations_file = os.path.join(val_dir, "val_annotations.txt")

print("Organizing validation images...")
with open(annotations_file) as f:
    for line in f:
        fn, cls, *_ = line.split("\t")
        class_dir = os.path.join(val_dir, cls)
        os.makedirs(class_dir, exist_ok=True)
        
        src = os.path.join(val_dir, "images", fn)
        dst = os.path.join(class_dir, fn)
        shutil.copyfile(src, dst)

# Rimuove la cartella non più necessaria
shutil.rmtree(os.path.join(val_dir, "images"))

transform = T.Compose([
    T.Resize((224, 224)),  # Resize to fit the input dimensions of the network
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

tiny_imagenet_dataset_train = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/train', transform=transform)
tiny_imagenet_dataset_val = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/val', transform=transform)

train_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_train, batch_size=32, shuffle=True)
val_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_val, batch_size=32, shuffle=False)


