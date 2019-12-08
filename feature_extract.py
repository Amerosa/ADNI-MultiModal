import argparse
import os
from pathlib import Path

from torchvision import transforms, models
from torch.utils.data import Dataset, random_split, DataLoader
from data_loader import AdniDataset, show_batch
from torch import nn
import matplotlib.pyplot as plt
import torch

parser = argparse.ArgumentParser(description='ADNI Dataset Training')
parser.add_argument('caps', metavar='DIR', help='Folder that contains CAPS subjects compliant dataset')
parser.add_argument('tsv', metavar='FILE', help='File of the merged tsv files created by clinica pipeline')

args = parser.parse_args()

transform = transforms.Compose(
    [
        transforms.Resize((224,224), interpolation=0),
        transforms.ToTensor()
    ])


mri_dataset = AdniDataset(args.caps, mode='mri', transform=transform)
pet_dataset = AdniDataset(args.caps, mode='pet', transform=transform)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

train_size = int(0.6*len(mri_dataset))
val_size = test_size = int(0.2*len(mri_dataset))

#TODO Make sure to set a random seed that way I have the same splits each time!

mri_train_dataset, mri_test_dataset, mri_val_dataset = random_split(mri_dataset, [train_size, test_size, val_size])
pet_train_dataset, pet_test_dataset, pet_val_dataset = random_split(pet_dataset, [train_size, test_size, val_size])

mri_dataloaders = {
                'train': DataLoader(mri_train_dataset, batch_size=32),  
                'test': DataLoader(mri_test_dataset, batch_size=32),
                 'val': DataLoader(mri_val_dataset, batch_size=32)
                }

pet_dataloaders = {
                'train': DataLoader(pet_train_dataset, batch_size=32),  
                'test': DataLoader(pet_test_dataset, batch_size=32),
                 'val': DataLoader(pet_val_dataset, batch_size=32)
                }

dataset_sizes = {'train': train_size, 'test': test_size, 'val': val_size}

#msample, _ = next(iter(mri_dataloaders['train']))
#psample, _ = next(iter(pet_dataloaders['train']))

#print(msample.shape)
#print(psample.shape)

#show_batch(msample)
#show_batch(psample)

model_ft = models.resnet18(pretrained=True, progress=True)



ftr_extractor= nn.Sequential( *list(model_ft.children())[:-1] ) 
#for param in ftr_extactor.parameters():
#    param.requires_grad = False

ftr_extractor = ftr_extractor.to(device=device)

if not os.path.isdir(Path('./features')):
    os.mkdir(Path('./features'))


check = True
with torch.no_grad():
    for i, data in enumerate(mri_dataloaders['train']):
        mode = 'mri'
        phase = 'train'
        inputs, labels = data
        


        inputs = inputs.to(device=device, dtype=torch.float)
        labels = labels.to(device=device, dtype=torch.long)

        outputs = ftr_extractor(inputs)
       
        if check:
            _inputs = torch.empty(tuple(outputs.shape), device=device)
            _inputs = outputs
            check == False
        else:
            _inputs = torch.cat((_inputs, outputs), 0)

        
                

    print(_inputs.shape)
    #torch.save(features, Path('./features/' + mode + phase + thing + '.pt'))

