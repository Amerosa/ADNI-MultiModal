import os
import fnmatch
import nibabel as nib
from PIL import Image
import numpy as np
from pathlib import Path
from skimage.measure import shannon_entropy
import glob
import argparse
from numpy.random import default_rng


parser = argparse.ArgumentParser(description='Create a subset of data basing slices off entropy significance.')
parser.add_argument('dataset_root', metavar='DIR', help='Folder that contains the dataset you want to use')
parser.add_argument('threshold', type=int, help='The amount of important slices you want from the volume')

args = parser.parse_args()

def get_image(path):
    img = nib.load(path) #nii file type medical image
    return img.get_fdata() #Converted to a numpy array type
    
def shannon_max(image):
    entropies = [ (shannon_entropy(image[:,i,:]), i) for i in range(image.shape[1]) ]
    entropies = sorted(entropies, key=lambda x: x[0], reverse=True)
    return list(list(zip(*entropies[:args.threshold]))[1])

def shannon_split_extreme(image):
    entropies = [ (shannon_entropy(image[:,i,:]), i) for i in range(image.shape[1]) ]
    entropies = sorted(entropies, key=lambda x: x[0], reverse=True)
    return list(list(zip(*(entropies[:args.threshold//2] + entropies[-args.threshold//2:])))[1])

def shannon_split_weak(image):
    entropies = [ (shannon_entropy(image[:,i,:]), i) for i in range(image.shape[1]) ]
    entropies = sorted(entropies, key=lambda x: x[0], reverse=True)
    return list(list(zip(*(entropies[args.threshold//2:args.threshold] + entropies[-args.threshold//2:-args.threshold:-1])))[1])

def random_entropy(image):
    rng = default_rng() #Need this for non replcement of ints 
    return list(rng.choice(image.shape[1] - 1, size=args.threshold, replace=False)) 

def save_images(mri_destination, pet_destination, mri_image, pet_image, indicies):
   
    for sli in indicies:
        mri_temp = mri_image[:,sli,:]
        pet_temp = pet_image[:,sli,:]

        mri_temp = Image.fromarray( (mri_temp*255).astype(np.uint8) )
        pet_temp = Image.fromarray( (pet_temp*255).astype(np.uint8) )

        mri_temp.save(os.path.join(mri_destination, str(sli) + ".png"))
        pet_temp.save(os.path.join(pet_destination, str(sli) + ".png"))
    
def join_subject_lists(mri, pet):
    temp = []
    for i, subject in enumerate(mri):
        temp.append( (*subject, pet[i][2] ) ) #Indexing the same subject but getting the pet path
    return temp
        
def get_paths(dataset_root):
    mri_subjects = []
    pet_subjects = []

    for root, dirs, files in os.walk(dataset_root):
        for file in files:

            sub_id, session = file.split('_')[:2]
        
            if fnmatch.fnmatch(file, "*segm-graymatter*fwhm-8mm*"):
                mri_path = os.path.join(root, file )
                mri_subjects.append( (sub_id, session, mri_path) )

            if fnmatch.fnmatch(file, "*pet*fwhm-8mm*"):
                pet_path = os.path.join(root, file )
                pet_subjects.append( (sub_id, session, pet_path) )
                

    return join_subject_lists(mri_subjects, pet_subjects)

def create_subject_dir(subject, root):

    sub_id, session, _, _ = subject
    new_mri_path = os.path.join(root, sub_id, session, 'mri')
    new_pet_path = os.path.join(root, sub_id, session, 'pet')

    if not os.path.exists(new_mri_path):
        os.makedirs(new_mri_path)

    if not os.path.exists(new_pet_path):
        os.makedirs(new_pet_path)

    return new_mri_path, new_pet_path

def main():
    subjects = get_paths(args.dataset_root) 

    subsets = [ ("data/entropy_" + str(args.threshold), shannon_max),
                ("data/random_" + str(args.threshold), random_entropy),
                ("data/extreme_split_" + str(args.threshold), shannon_split_extreme),
                ("data/weak_split_" + str(args.threshold), shannon_split_weak) ]

    for subj_root, fn in subsets:

        print("Images will be store in", subj_root)            
        for subject in subjects:
            new_mri_path, new_pet_path = create_subject_dir(subject, subj_root)

            mri_image = get_image(subject[2])
            pet_image = get_image(subject[3])

            indicies = fn(mri_image)
            
            save_images(new_mri_path, new_pet_path, mri_image, pet_image, indicies)

main()
print()
print("Finished Creating Subsets!")