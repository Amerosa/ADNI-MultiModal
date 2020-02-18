import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import copy
import argparse
from pathlib import Path

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class TensorDataset(torch.utils.data.TensorDataset):
    def __init__(self, mri_features, pet_features, labels):
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

naive_clf.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(naive_clf.parameters(), lr=0.001, weight_decay=0.01)

def train(model, criterion, optimizer, num_epochs=25):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch: {epoch+1}')
        for phase in ['train', 'val']:

            running_loss = 0.0
            running_corrects = 0.0

            for batch_idx, (inputs,labels) in enumerate(dataloaders[phase]):    
                
                #inputs = torch.t(inputs)
                inputs = inputs.to(device)
                #labels = labels.squeeze(1)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    #print(inputs.shape)
                    #print(labels.shape)
                    outputs = model(inputs)
                    #print(outputs.shape)
                    #outputs = outputs.reshape((outputs.size(0), 4))
                    #print(outputs.shape)
                    #print(labels.shape)
                    _, preds = torch.max(outputs, 1)
                    #print(preds.shape)
                    #preds = preds.reshape(-1)
                    #print(outputs)
                    #print(preds.shape)
                    #print(preds)
                    #print(vals)
                    loss = criterion(outputs, labels)
                
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc*100))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
    
    print()
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val acc: {:.4f}'.format(best_acc*100))

    model.load_state_dict(best_model_wts)
    
    return model

def testing(model):
    running_corrects = 0.0
    with torch.no_grad():
        for batch_idx, (inputs, labels) in enumerate(dataloaders['test']):
            #inputs = inputs.transpose(1,3)
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            #outputs = outputs.reshape((outputs.size(0), 4))
            _, preds = torch.max(outputs, 1)

            running_corrects += torch.sum(preds == labels)
            #print(running_corrects)
    print()
    print('Test Acc: {:.4f} '.format( (running_corrects / dataset_sizes['test'])*100 ))

#print(dataset_sizes['test'])
#trained_model = train(net, criterion, optimizer, 100)
#torch.save(trained_model, 'features/baseline.pt')
#testing(trained_model)
parser = argparse.ArgumentParser()
parser.add_argument('features', metavar='DIR', help='Directory of the feature tensors extracted')
parser.add_argument('labels', metavar='DIR', help='Directory of the labels')

args = parser.parse_args()
ft_dir = Path(args.features)
lb_dir = Path(args.labels)
dataloaders = {}

#TODO Super messy way to load all the data, may need to fix this in the future!
train_set = TensorDataset(torch.load(ft_dir / 'mri-train-features.pt'), torch.load(ft_dir / 'pet-train-features.pt'), torch.load(lb_dir / 'train-labels.pt') )
val_set   = TensorDataset(torch.load(ft_dir / 'mri-val-features.pt'), torch.load(ft_dir / 'pet-val-features.pt'), torch.load(lb_dir / 'val-labels.pt') )
test_set  = TensorDataset(torch.load(ft_dir / 'mri-test-features.pt'), torch.load(ft_dir / 'pet-test-features.pt'), torch.load(lb_dir / 'test-labels.pt') )

dataset_sizes = { 'train': len(train_set), 'val'  : len(val_set), 'test' : len(test_set) }

dataloaders = {
                'train' : DataLoader(train_set, batch_size=128, shuffle=False),
                'val' : DataLoader(val_set, batch_size=128, shuffle=False),
                'test' : DataLoader(test_set, batch_size=128, shuffle=False)
}

#sample, label = next(iter(dataloaders['test']))
#print(sample.shape, label.shape)

trained_model = train(naive_clf, criterion, optimizer, 100)
testing(trained_model)
