import argparse
import os
from pathlib import Path

from torchvision import transforms, models
from torch.utils.data import Dataset, random_split, DataLoader, ConcatDataset
from torch import nn
import matplotlib.pyplot as plt
import torch
from  torch.optim import lr_scheduler
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

from models.nets import *
from collections import defaultdict

parser = argparse.ArgumentParser(description='ADNI Dataset Training')
parser.add_argument('data', metavar='DIR', help='Folder that contains the dataset you want to use')
parser.add_argument('tsv', metavar='FILE', help='Labels')

args = parser.parse_args()

#TODO Need to fix the file path thingy so that it can accept a trailing \ otherwise the whole thing breaks 
subset = args.data.split('\\')[-1]
destination_dir = os.path.join('features', subset)
model_destination = os.path.join('models', subset + '.pth')

if not os.path.exists(destination_dir):
    os.mkdir(destination_dir)

print(f'Destination to save features is {destination_dir} and model weights will be saved in {model_destination}')

class ADNI(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        
        self.threshold = int(root_dir.replace('\\', '').split('_')[-1])
        self.root = root_dir
        self.csv = pd.read_csv(csv_path, sep='\t')
        self.paths = self.make_paths()
        self.transform = transform
        
        self.class_to_num = {'CN':0, 'nMCI':1, 'cMCI':2, 'AD':3}
        
    def __len__(self):
        #print(self.paths)
        return len(self.paths)
    
    def __getitem__(self, index):
        
        subject = self.csv.iloc[index//self.threshold]
        
        mri_image = Image.open(self.paths[index][0])
        pet_image = Image.open(self.paths[index][1])
        
        
        #Applyin transformations if any
        #Then copying the grayscale image across 3 channels for Resenet
        
        if transform:
            mri_image = self.transform(mri_image).repeat_interleave(3, dim=0)
            pet_image = self.transform(pet_image).repeat_interleave(3, dim=0)
        
        return {"id": subject["participant_id"], 
                "sess": subject["session_id"], 
                "mri": mri_image, 
                "pet": pet_image, 
                "class": self.class_to_num[subject["diagnosis"]]}
    
    def make_paths(self):
        all_paths = []
        for root, dirs, files in os.walk(self.root):
            for file in files:
                all_paths.append(os.path.join(root, file))
        all_paths = list(zip(*[iter(all_paths)]*self.threshold)) #Break into chunks of threshold size
        mri_paths = [item for tup in all_paths[::2] for item in tup]
        pet_paths = [item for tup in all_paths[1::2] for item in tup]
        return list(zip(mri_paths, pet_paths))

def naive_feature_extraction(dataloader):
    model = ResnetNaive().to(device).eval()
    features = defaultdict(list)
    with torch.no_grad():
        for idx, batch in enumerate(dataloader, start=1):               
            mri_inputs, pet_inputs = batch["mri"].to(device), batch["pet"].to(device)
            mri_outputs, pet_outputs = model(mri_inputs, pet_inputs)
            
            print(f'Batch: {idx:03d}, Mri shape: {mri_outputs.shape}, Pet shape: {pet_outputs.shape}')
            
            features['mri'].append(mri_outputs.squeeze().cpu())
            features['pet'].append(pet_outputs.squeeze().cpu())
            features['class'].append(batch["class"])
            
    print(f'Saving to, {destination_dir} ...')
    features = { mode: torch.cat(feats, dim=0) for mode, feats in features.items() }

    print(f"Mri features: {features['mri'].shape} Pet features: {features['pet'].shape} Labels: {features['class'].shape}")
    
    torch.save(features['mri'], os.path.join(destination_dir, 'naive_mri_features.pt'))
    torch.save(features['pet'], os.path.join(destination_dir, 'naive_pet_features.pt'))
    torch.save(features['class'], os.path.join(destination_dir, 'naive_labels.pt'))
    print('Naive features have been extracted and saved!')
    
def train_multimodal(dataloader, num_epochs=25):
    since = time.time()

    model = ResnetMultiModal().to(device)
    criterion = nn.CrossEntropyLoss()
    mri_optimizer = optim.SGD(model.mri.parameters(), lr=0.0001, weight_decay=0.01, momentum=0.9)
    pet_optimizer = optim.SGD(model.pet.parameters(), lr=0.0001, weight_decay=0.01, momentum=0.9)

    best_mri_model_wts = copy.deepcopy(model.mri.state_dict())
    best_pet_model_wts = copy.deepcopy(model.pet.state_dict())
    mri_best_acc = 0.0
    pet_best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}'.format(epoch+1))
        print('-'*20)

        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()
            else:
                model.eval()

            mri_running_loss = 0.0
            mri_running_corrects = 0.0

            pet_running_loss = 0.0
            pet_running_corrects = 0.0            

            for batch_idx, subject in enumerate(dataloader[phase]):
                
                mri_in, pet_in, labels = subject['mri'].to(device), subject['pet'].to(device), subject['class'].to(device)

                mri_optimizer.zero_grad()
                pet_optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    mri_out, pet_out = model(mri_in, pet_in)
                    _, mri_preds = torch.max(mri_out, 1)
                    _, pet_preds = torch.max(pet_out, 1)
                    mri_loss, pet_loss = criterion(mri_out, labels), criterion(pet_out, labels)

                    if phase == 'train':
                        mri_loss.backward()
                        mri_optimizer.step()

                        pet_loss.backward()
                        pet_optimizer.step()
                
                mri_running_loss += mri_loss.item() * mri_in.size(0)
                mri_running_corrects += torch.sum(mri_preds == labels)

                pet_running_loss += pet_loss.item() * pet_in.size(0)
                pet_running_corrects += torch.sum(pet_preds == labels)
                #print('Batch {} done...'.format(batch_idx))

            mri_epoch_loss = mri_running_loss / dataset_sizes[phase]*100
            mri_epoch_acc = mri_running_corrects.double() / dataset_sizes[phase]*100

            pet_epoch_loss = pet_running_loss / dataset_sizes[phase]*100
            pet_epoch_acc = pet_running_corrects.double() / dataset_sizes[phase]*100

            print('Mri {} Loss: {:.4f} Acc: {:.4f}'.format(phase, mri_epoch_loss, mri_epoch_acc))
            print('Pet {} Loss: {:.4f} Acc: {:.4f}'.format(phase, pet_epoch_loss, pet_epoch_acc))

            if phase == 'val' and mri_epoch_acc > mri_best_acc:
                mri_best_acc = mri_epoch_acc
                mri_best_model_wts = copy.deepcopy(model.mri.state_dict())

            if phase == 'val' and pet_epoch_acc > pet_best_acc:
                pet_best_acc = pet_epoch_acc
                pet_best_model_wts = copy.deepcopy(model.pet.state_dict())
    
    print()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best Mri val Acc: {:.4f}'.format(mri_best_acc))
    print('Best Pet val Acc: {:.4f}'.format(pet_best_acc))
    print()

    model.mri.load_state_dict(mri_best_model_wts)
    model.pet.load_state_dict(pet_best_model_wts)

    torch.save(model, model_destination)

    return model

def trained_feature_extraction(model, dataloader): 
    model.feature_extractor_mode()
    model = model.to(device).eval()
    features = defaultdict(list)
    with torch.no_grad():
        for idx, batch in enumerate(dataloader, start=1):               
            mri_inputs, pet_inputs = batch["mri"].to(device), batch["pet"].to(device)
            mri_outputs, pet_outputs = model.mri(mri_inputs), model.pet(pet_inputs)

            print(f'Batch: {idx:03d}, Mri shape: {mri_outputs.shape}, Pet shape: {pet_outputs.shape}')

            features['mri'].append(mri_outputs.squeeze().cpu())
            features['pet'].append(pet_outputs.squeeze().cpu())
            features['class'].append(batch["class"])
    
    print(f'Saving to, {destination_dir} ...')
    features = { mode: torch.cat(feats, dim=0) for mode, feats in features.items() }

    print(f"Mri features: {features['mri'].shape} Pet features: {features['pet'].shape} Labels: {features['class'].shape}")
    torch.save(features['mri'], os.path.join(destination_dir, 'trained_mri_features.pt'))
    torch.save(features['pet'], os.path.join(destination_dir, 'trained_pet_features.pt'))
    torch.save(features['class'], os.path.join(destination_dir, 'trained_labels.pt'))
    print('Trained features have been extracted and saved!')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize((224,224), interpolation=0),
        transforms.ToTensor()
    ])

print("Loading Dataset...")
adni_dataset = ADNI(args.tsv, args.data, transform=transform)
print(f'Adni dataset has {len(adni_dataset)} samples')

train_size = int(0.6*len(adni_dataset))
val_size = test_size = int(0.2*len(adni_dataset))
dataset_sizes = {'train': train_size, 'val': val_size, 'test': test_size}

print(f"Training size: {dataset_sizes['train']} | Validation size: {dataset_sizes['val']} | Testing size: {dataset_sizes['test']}")

torch.manual_seed(42)
adni_trainset, adni_valset, adni_testset = random_split(adni_dataset, dataset_sizes.values())

adni_dataloaders = {
            'train': DataLoader(adni_trainset, batch_size=128),
            'val': DataLoader(adni_valset, batch_size=128),
            'test': DataLoader(adni_testset, batch_size=128)
}

# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.model_selection import cross_val_score, cross_val_predict
# from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
# from sklearn.svm import LinearSVC


# print("Classifying...")
# clf = LinearSVC(dual=False) # number of samples greater than the features so dual needs to be false as per scikits docs
# cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
# #TODO corss val predict is not working becuase it only works on partitions?? Try stratified kfold instead 

# scores = cross_val_score(clf, features, labels, cv=cv)

# print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# extract_features(mri_model, mri_dataloader, 'mri')
# extract_features(pet_model, pet_dataloader, 'pet')
# serialize_labels()

naive_feature_extraction( DataLoader(ConcatDataset([adni_trainset, adni_valset, adni_testset]), batch_size=128) )
trained_model = train_multimodal(adni_dataloaders, num_epochs=1)
trained_feature_extraction( trained_model, DataLoader(ConcatDataset([adni_trainset, adni_valset, adni_testset]), batch_size=128) )  














