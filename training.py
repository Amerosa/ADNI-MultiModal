import torch
from torch.utils.data import DataLoader, random_split, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import copy
import argparse
from pathlib import Path
import os 
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, cross_val_predict, StratifiedShuffleSplit
import csv

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TensorDataset(torch.utils.data.TensorDataset):
    def __init__(self, naive_features, trained_features, labels):
        super(TensorDataset).__init__()
        self.mri = mri_features.squeeze() #[samples x features x 1 x 1] => [samples x features]
        self.pet = pet_features.squeeze() 
        self.labels = labels #[samples x 1]

    def __len__(self):
        return self.labels.shape[0]
    
    def __getitem__(self, idx):
        #[1 x feautres]                  
        temp = torch.cat((self.mri[idx], self.pet[idx]), 0) 
        return (temp, self.labels[idx])

class FullyConnectedClf(nn.Module):
    def __init__(self):
        super(FullyConnectedClf, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 4)

    def forward(self , x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
   
        return x 

naive_clf = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512,4)           
)

def classify(feature_dict):
    clf = LinearSVC(dual=False)
    cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2)
    scores = cross_val_score(clf, 
                        torch.cat( (feature_dict['mri'], feature_dict['pet']), dim=1).numpy(), 
                        feature_dict['class'].numpy(), 
                        cv=cv)
    return scores.mean(), scores.std()




#print( torch.cat( (naive_features['mri'], naive_features['pet']), dim=1 ).numpy() )

with open('results.csv', mode='w') as results_file:
    results_writer = csv.writer(results_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

    for experiment in ['entropy_30', 'random_30', 'extreme_split_30', 'weak_split_30']:
        
        feature_dir = os.path.join('features', experiment)
        
        naive_features = {'mri': torch.load(os.path.join(feature_dir, 'naive_mri_features.pt')),
                    'pet': torch.load(os.path.join(feature_dir, 'naive_pet_features.pt')),
                    'class': torch.load(os.path.join(feature_dir, 'naive_labels.pt'))
                    }

        trained_features = {'mri': torch.load(os.path.join(feature_dir, 'trained_mri_features.pt')),
                    'pet': torch.load(os.path.join(feature_dir, 'trained_pet_features.pt')),
                    'class': torch.load(os.path.join(feature_dir, 'trained_labels.pt'))
                    }

        results_writer.writerow([experiment, 'naive', 'cat', *classify(naive_features)])
        results_writer.writerow([experiment, 'trained', 'cat', *classify(trained_features)])

#print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))