import os
import glob
import pandas as pd
import numpy as np
import nibabel as nib 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, random_split, DataLoader
from torchvision import transforms, models
import torchvision
import torch
from torch import nn, optim
import fnmatch
import copy
import time
#from image_slices_viewer import IndexTracker

#TODO Needo to fix the file that this comes from
#https://matplotlib.org/gallery/animation/image_slices_viewer.html

def make_paths(root_dir, pattern):
    file_paths = []
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for file in files:
            if fnmatch.fnmatch(file, pattern):
                file_paths.append( (os.path.join(root, file)) )
    
    return file_paths


class IndexTracker(object):
    def __init__(self, ax, X):
        self.ax = ax
        ax.set_title('use scroll wheel to navigate images')

        self.X = X
        rows, cols, self.slices = X.shape
        self.ind = self.slices//2

        self.im = ax.imshow(self.X[:, self.ind, :])
        self.update()

    def onscroll(self, event):
        #print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[:, self.ind, :])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

def view_full_image(sample):
    fig, ax = plt.subplots(1,1)
    tracker = IndexTracker(ax, sample)

    fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
    plt.show()

#sample_path = "../caps_dir/subjects/sub-ADNI002S0295/ses-M00/t1/spm/segmentation/normalized_space"
#sample_image = "sub-ADNI002S0295_ses-M00_T1w_segm-graymatter_space-Ixi549Space_modulated-off_probability.nii.gz"
#images = glob.glob(root)
#example = os.path.join(sample_path, sample_image)
#image = nib.load(example)
#header = image.header
#print(header)

#image_data = image.get_fdata()
#print(image_data.shape)

#fig, ax = plt.subplots(1,1)
#tracker = IndexTracker(ax, image_data)
#fig.canvas.mpl_connect('scroll_event', tracker.onscroll)
#plt.show()


#plt.imshow(image_data[:,:,60], cmap="gray")
#plt.show()


#adni_merge.set_index['participant']    
def convert_subjects_format(s):
    s = s[8:]
    s = s[:3] + '_' + s[3] + '_' + s[4:]
    return s

label_to_num = {"CN":0, "AD":1, "LMCI":2}
num_to_label = {0:"CN", 1:"AD", 2:"LMCI"}

class GreymatterDataset(Dataset):
    """Dataset of the graymatter normalized space segmenation"""

    def __init__(self, root_dir, transform=None):
        
        self.num_slices = 145
        pattern = "*_T1w_segm-graymatter_space-Ixi549Space_modulated-off_probability.nii.gz"
        self.paths = make_paths(root_dir, pattern)
        self.paths.sort()

        #This is turing the master list of subjects into a dict with the labels
        #adni_merge = pd.read_csv('comprehensive_study.csv')
        #adni_merge = adni_merge[['Subject', 'Group']]
        #adni_merge.drop_duplicates(inplace=True)
        #adni_merge = pd.Series(adni_merge['Group'].values, index=adni_merge['Subject']).to_dict()

        #This is only the patients used and we are mathing it with the master list to generate labels
        #patients = pd.read_csv('subject_visit.csv', sep='\t')
        #patients = patients['participant_id']
        #patients = patients.apply(convert_subjects_format)

        classes = pd.read_csv('../merge-tsv.tsv', sep='\t')
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
        image = image.get_fdata()
        image = np.expand_dims(image, axis=0)

        #print(self.paths[path_idx], self.labels[path_idx], "slice:", s)
        
        return (image[:,:,s,:], self.labels[path_idx])

    def print_paths(self):
        for i, path in enumerate(self.paths):
            print(path, num_to_label[self.labels[i]])

dataset= GreymatterDataset('../caps_dir/subjects', transform=transforms.ToTensor())

#sample, label = dataset.__getitem__(256789)

#plt.imshow(sample, cmap='gray')
#plt.show()

train_size = int(0.6*len(dataset))
val_size = test_size = int(0.2*len(dataset))

train_dataset, test_dataset, val_dataset = random_split(dataset, [train_size, test_size, val_size])

dataloaders = {
                'train': DataLoader(train_dataset, batch_size=32),  
                'test': DataLoader(test_dataset, batch_size=32),
                 'val': DataLoader(val_dataset, batch_size=32)
              }

dataset_sizes = {'train': train_size, 'test': test_size, 'val': val_size}


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def train_model(model, criterion, optimizer, num_epochs=100):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device=device, dtype=torch.float)
                labels = labels.to(device=device, dtype=torch.long)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./cifar10', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./cifar10', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')



def train(model, criterion, optimizer, num_epochs=25):
    for epoch in range(num_epochs):
        running_loss = 0.0
        
        print('-'*10)
        print(f"Epoch: {epoch+1}/{num_epochs}")
        print('-'*10)
        
        for i, data in enumerate(trainloader):
            inputs, labels = data
            model.train()

            inputs = inputs.to(device=device, dtype=torch.float)
            labels = labels.to(device=device, dtype=torch.long)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        
            running_loss += loss.item()
            
            if i % 2000 == 1999:
                print("[MiniBatch {}] loss: {}".format(i+1, running_loss/2000))
                running_loss = 0.0
        print()
    print("Finished Training")            

def show_image_batch(sample, rows, cols ):
    images, labels = sample
    print(labels)
    fig = plt.figure(figsize=(50,50))
    for i in range(rows*cols):
        sub = fig.add_subplot(rows, cols, i+1)
        sub.imshow(images[i,0,:,:])
        sub.text(2, 2, num_to_label[labels[i].item()], bbox={'facecolor': 'white', 'pad': 5})
        sub.axes.get_xaxis().set_visible(False)
        sub.axes.get_yaxis().set_visible(False)
    plt.axis('off')
    plt.show()


#samples = [dataset.__getitem__(i) for i in range(0,141,10) ]
# for sample in samples :
#     image, _ = sample
#     plt.imshow(image[0,:,:])
#     plt.show()
#samples = [dataset.__getitem__(i) for i in range(290)]



model_ft = models.resnet18(pretrained=False, progress=True)
#model_ft.features._modules['0'] = nn.Conv2d(1, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
#model_ft.classifier._modules['6'] = nn.Linear(4096, 10)
model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
model_ft.fc = nn.Linear(in_features=512, out_features=3, bias=True)



model_ft = model_ft.to(device)

criterion = nn.CrossEntropyLoss()

optimizer_ft = optim.Adam(model_ft.parameters(), lr=0.001)

model_ft = train_model(model_ft, criterion, optimizer_ft, num_epochs=25)

correct = 0
total = 0
with torch.no_grad():
    for data in dataloaders['test']:
        images, labels = data
        outputs = model_ft(images)
        _, preds = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

print(f"Accuracy of the netowrk is {(correct/total) * 100}")


