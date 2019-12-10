import argparse
import os
from pathlib import Path

from torchvision import transforms, models
from torch.utils.data import Dataset, random_split, DataLoader
from torch import nn
import matplotlib.pyplot as plt
import torch

import argparse
import os
import glob
import pandas as pd
import numpy as np
import nibabel as nib 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision.utils import make_grid
from torchvision import transforms, models
import torchvision
import torch
from torch import nn, optim
import fnmatch
import copy
import time
from PIL import Image

parser = argparse.ArgumentParser(description='ADNI Dataset Training')
parser.add_argument('caps', metavar='DIR', help='Folder that contains CAPS subjects compliant dataset')
parser.add_argument('tsv', metavar='FILE', help='File of the merged tsv files created by clinica pipeline')

args = parser.parse_args()

def make_paths(root_dir, pattern):
    file_paths = []
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for file in files:
            if fnmatch.fnmatch(file, pattern):
                file_paths.append( (os.path.join(root, file)) )
    
    return file_paths

label_to_num = {"CN":0, "AD":1, "LMCI":2}
class AdniDataset(Dataset):
    """Basic class for ADNI Caps compliant directory structure,
    just change the pattern for which type of image modality you want"""

    def __init__(self, root_dir, mode='mri', transform=None):
        
        pattern = None

        if mode == 'mri':
            pattern = "*_T1w_segm-graymatter_space-Ixi549Space_modulated-on_fwhm-8mm_probability.nii.gz"
        
        if mode == 'pet':
            pattern = "*_task-rest_acq-fdg_pet_space-Ixi549Space_suvr-pons_mask-brain_fwhm-8mm_pet.nii.gz"

        assert pattern, "Bad pattern matching generation check modality!"

        self.num_slices = 145
        #pattern = "*_T1w_segm-graymatter_space-Ixi549Space_modulated-off_probability.nii.gz"
        
        self.paths = make_paths(root_dir, pattern)
        self.paths.sort()

        classes = pd.read_csv(args.tsv, sep='\t')
        classes = classes['diagnosis_bl'].tolist()
        #print(classes)

        self.labels = []
        for cl in classes:
            self.labels.append(label_to_num[cl])
        
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.paths) * self.num_slices

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        s = idx % self.num_slices
        path_idx = idx // self.num_slices
    

        image = nib.load(self.paths[path_idx])
        #print(f'{self.paths[path_idx]} \t {self.labels[path_idx]} \t {s}')
        image = image.get_fdata()
        #print("Original Shape", image.shape)
        image = image[:,s,:]
        #image = np.expand_dims(image, axis=0)
        
        image = Image.fromarray(image)
        #plt.imshow(image)
        #plt.show()

        #print("Convert to PIL", image.size)

        if self.transform:
            image = self.transform(image)

        image = image.repeat_interleave(3, dim=0)
        #print("Post Transformation", image.shape)

        #print("Before Sample", image.shape)

        return (image, self.labels[path_idx])


transform = transforms.Compose(
    [
        transforms.Resize((224,224), interpolation=0),
        transforms.ToTensor()
    ])

mri_dataset = AdniDataset(args.caps, mode='mri', transform=transform)
pet_dataset = AdniDataset(args.caps, mode='pet', transform=transform)

dataloader = {
            'mri': DataLoader(mri_dataset, batch_size=128),
            'pet': DataLoader(pet_dataset, batch_size=128)
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

extractor = models.resnet18(pretrained=True, progress=True)
extractor = nn.Sequential( *list(extractor.children())[:-1] ) 
extractor = extractor.to(device)

extractor.eval()
print(device)
with torch.no_grad():
    for mode in ['mri', 'pet']:
        print(f"Extracting {mode} features")
        for idx, batch in enumerate(dataloader[mode]):
            inputs, _ = batch
            inputs = inputs.to(device, dtype=torch.float)
            #labels = labels.to(device=device, dtype=torch.long)

            outputs = extractor(inputs)

            if idx == 0:
                features = outputs
                continue

            features = torch.cat((features, outputs), 0)

        print(f'{mode}: {features.shape}')
        print("Saving features in file...")
        torch.save(features, Path('./features/' + mode + '-feautres.pt'))
        print("File Saved")

print("Concat the labels")
for idx, (_, labels) in enumerate(dataloader['mri']):

    if idx == 0:
        acc = labels
        continue

    acc = torch.cat((acc,labels), 0)       

print(f'Labels: {acc.shape}')
print("Saving labels in file...")
torch.save(acc, Path('./features/labels.pt'))
print("File Saved")
