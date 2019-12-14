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
parser.add_argument('tsv', metavar='FILE', help='Labels')
parser.add_argument('--training', dest='training', action='store_true')
parser.add_argument('--no-training', dest='training', action='store_false')
parser.set_defaults(training=True)

args = parser.parse_args()

def make_paths(root_dir, pattern):
    file_paths = []
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for file in files:
            if fnmatch.fnmatch(file, pattern):
                file_paths.append( (os.path.join(root, file)) )
    
    return file_paths

label_to_num = {'CN':0, 'nMCI':1, 'cMCI':2, 'AD':3}

#TODO I can bundle the modes into one dataset object instead of having two of them currently
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

        classes = pd.read_csv(args.tsv, sep='\t', usecols=['diagnosis'])

        self.labels = []
        for cl in classes['diagnosis']:
            self.labels.append(label_to_num[cl])
        
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

        image = image.repeat_interleave(3, dim=0) #Convert grayscale to "RGB" by copying values across 3 dim
        #print("Post Transformation", image.shape)

        #print("Before Sample", image.shape)

        return (image, self.labels[path_idx])

def train(model, criterion, optimizer, dataloader, num_epochs=100):
    
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}'.format(epoch+1))

        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0.0

            for batch_idx, (inputs, labels) in enumerate(dataloader[phase]):
                
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
                #print('Batch {} done...'.format(batch_idx))

            epoch_loss = running_loss / dataset_sizes[phase]*100
            epoch_acc = running_corrects.double() / dataset_sizes[phase]*100

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
    print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:.4f}'.format(best_acc))
    print()

    model.load_state_dict(best_model_wts)
    return model

def extract_features(model, dataloader, mode):
    model = model.eval()
    
    if not Path('./features').is_dir():
        os.mkdir(Path('./features'))

    with torch.no_grad():
        for phase in ['train', 'val', 'test']:
            print(f"Extracting {mode} {phase} features")
            for idx, batch in enumerate(dataloader[phase]):
               
                inputs, _ = batch
                inputs = inputs.to(device)
                outputs = model(inputs)

                if idx == 0:
                    features = outputs
                    continue

                features = torch.cat((features, outputs), 0)

            print(f'{mode}: {features.shape}')
            print("Saving features in file...")
            if args.training:
                if not Path('./features/trained').is_dir():
                    os.mkdir(Path('./features/trained'))
                torch.save(features, Path('./features/trained/' + mode + '-' + phase + '-features.pt'))
            else:
                if not Path('./features/not_trained').is_dir():
                    os.mkdir(Path('./features/not_trained'))            
                torch.save(features, Path('./features/not_trained/' + mode + '-' + phase + '-features.pt'))
            print("File Saved")

def serialize_labels():
    print("Concat the labels")
    for phase in ['train', 'val', 'test']:
        for idx, (_, labels) in enumerate(mri_dataloader[phase]):

            if idx == 0:
                acc = labels
                continue

            acc = torch.cat((acc,labels), 0)       

        print(f'Labels: {acc.shape}')
        print("Saving labels in file...")
        torch.save(acc, Path('./features/labels.pt'))
        print("File Saved")

transform = transforms.Compose(
    [
        transforms.Resize((224,224), interpolation=0),
        transforms.ToTensor()
    ])

mri_dataset = AdniDataset(args.caps, mode='mri', transform=transform)
pet_dataset = AdniDataset(args.caps, mode='pet', transform=transform)

train_size = int(0.6*len(mri_dataset))
val_size = test_size = int(0.2*len(mri_dataset))

torch.manual_seed(42)
mri_trainset, mri_valset, mri_testset = random_split(mri_dataset, [train_size,val_size,test_size])

torch.manual_seed(42)
pet_trainset, pet_valset, pet_testset = random_split(pet_dataset, [train_size,val_size,test_size])

dataset_sizes = {'train': train_size, 'val': val_size, 'test': test_size}

mri_dataloader = {
            'train': DataLoader(mri_trainset, batch_size=128),
            'val': DataLoader(mri_valset, batch_size=128),
            'test': DataLoader(mri_testset, batch_size=128)
}

pet_dataloader = {
            'train': DataLoader(pet_trainset, batch_size=128),
            'val': DataLoader(pet_valset, batch_size=128),
            'test': DataLoader(pet_testset, batch_size=128)
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mri_model = models.resnet18(pretrained=True, progress=True)
pet_model = models.resnet18(pretrained=True, progress=True)

num_ftrs = mri_model.fc.in_features
mri_model.fc = nn.Linear(num_ftrs, 4)
pet_model.fc = nn.Linear(num_ftrs, 4)

if args.training:

    mri_model = mri_model.to(device)
    pet_model = pet_model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    mri_optimizer = optim.Adam(mri_model.parameters())
    pet_optimizer = optim.Adam(pet_model.parameters())

    mri_model = train(mri_model, criterion, mri_optimizer, mri_dataloader, num_epochs=25)
    pet_model = train(pet_model, criterion, pet_optimizer, pet_dataloader, num_epochs=25)

mri_model = nn.Sequential( *list(mri_model.children())[:-1] )
mri_model = mri_model.to(device)
pet_model = nn.Sequential( *list(pet_model.children())[:-1] )  
pet_model = pet_model.to(device)

extract_features(mri_model, mri_dataloader, 'mri')
extract_features(pet_model, pet_dataloader, 'pet')
serialize_labels()















