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
parser.add_argument('threshold', type=int, help='The amount of important slices you want from the volume')
parser.add_argument('--random', default=False, action='store_true', help='Enable this to use for random slice indexing instead of entropy')
args = parser.parse_args()

dataset_root = "F:/caps/subjects/"

if args.random:
    master_root = "F:/temp/" + "random_" + str(args.threshold)
else:
    master_root = "F:/temp/" + "entropy_" + str(args.threshold)

print("Images will be store in", master_root)

def get_image(path):
    img = nib.load(path) #nii file type medical image
    return img.get_fdata() #Converted to a numpy array type
    
def shannon(image):
    entropies = [ (shannon_entropy(image[:,i,:]), i) for i in range(image.shape[1]) ]
    entropies = sorted(entropies, key=lambda x: x[0], reverse=True)
    return list(zip(*entropies[:args.threshold]))

def random_entropy(num_slices):
    rng = default_rng() #Need this for non replcement of ints 
    return list(rng.choice(num_slices, size=args.threshold, replace=False)) 

def save_images(mri_destination, pet_destination, mri_image_path, pet_image_path):
   
    mri_image = get_image(mri_image_path)
    pet_image = get_image(pet_image_path)

    if args.random:
        indicies = random_entropy(mri_image.shape[1] - 1) #145 slices remember to sub 1 for indicies
    else:
        indicies = list(shannon(mri_image)[1])

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

def create_subject_dir(subject):

    sub_id, session, _, _ = subject
    new_mri_path = os.path.join(master_root, sub_id, session, 'mri')
    new_pet_path = os.path.join(master_root, sub_id, session, 'pet')

    if not os.path.exists(new_mri_path):
        os.makedirs(new_mri_path)

    if not os.path.exists(new_pet_path):
        os.makedirs(new_pet_path)

    return new_mri_path, new_pet_path

def main():
    subjects = get_paths(dataset_root)
    for subject in subjects:
        mri_path, pet_path = create_subject_dir(subject)
        save_images(mri_path, pet_path, subject[2], subject[3])

main()
print()
print("Finished Creating Subset!")