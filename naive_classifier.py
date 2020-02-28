import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms, models
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.svm import LinearSVC
import numpy as np 
import matplotlib.pyplot as plt

class ADNI(Dataset):
    def __init__(self, csv_path, root_dir, transform=None):
        
        self.threshold = int(root_dir.split('_')[-1])
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

def extract_features(model, dataloader):
    
    model = model.eval()
    features = []
    labels = []
    with torch.no_grad():
        for idx, batch in enumerate(dataloader):               
            mri_inputs = batch["mri"].to(device)
            pet_inputs = batch["pet"].to(device)
            #print("Batch: ", idx, "Mri shape: ", mri_inputs.shape, "Pet shape: ", pet_inputs.shape)
            #print(torch.max(mri_inputs))
            mri_outputs = model(mri_inputs).squeeze()
            pet_outputs = model(pet_inputs).squeeze()

            temp_result = torch.cat((mri_outputs, pet_outputs), 1) #Concat the two features into ==> [sample x features] : [4 x 1024 (512*2)]
            features.append(temp_result)
            
            labels.append(batch["class"])

    print("Features extracted!")
    return torch.cat(features, dim=0), torch.cat(labels, dim=0)  #Stack the list of accumulated batches of feautres into one big matrix to use for SVM

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose(
    [
        transforms.Resize((224,224), interpolation=0),
        transforms.ToTensor()
    ])

print("Loading Dataset...")
adni = ADNI("labels.tsv", "F:/temp/entropy_30", transform=transform)

#print("Length of the data set is: ", len(adni))

adni_loader = DataLoader(adni, batch_size=64, shuffle=False)

resnet = models.resnet18(pretrained=True, progress=True)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, 4)
resnet = nn.Sequential( *list(resnet.children())[:-1] ) #Remove the last fully connected layer for features
resnet = resnet.to(device)

features, labels = extract_features(resnet, adni_loader)
#print(features.shape)
#print(labels.shape)
features = features.cpu().numpy()
labels = labels.cpu().numpy()


print("Classifying...")
clf = LinearSVC(dual=False) # number of samples greater than the features so dual needs to be false as per scikits docs
cv = StratifiedShuffleSplit(n_splits=10, test_size=0.2)
#TODO corss val predict is not working becuase it only works on partitions?? Try stratified kfold instead 

scores = cross_val_score(clf, features, labels, cv=cv)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

#target_names = ["CN", "nMCI", "cMCI", "AD"]
#print("Accuracy: ", accuracy_score(labels, pred))
#print(classification_report(labels, pred, target_names=target_names))

#cfm = confusion_matrix(labels, pred)
#plt.matshow(cfm)
#plt.show()
 

