import os
import fnmatch
import nibabel as nib
from PIL import Image
import numpy as np
from pathlib import Path
from skimage.measure import shannon_entropy 


dataset_root = "F:/caps/subjects/"

subjects = os.listdir("F:/caps/subjects/")
print(len(subjects))


#For each subject we need to create a file with thier name and then create to child dirs mri and pet
#Then we populate with the entropy thresholds and random
#For one subject we need to grab the right image and process it with nibabel

def make_paths(root_dir, pattern):
    file_paths = []
    for root, dirs, files in os.walk(root_dir, topdown=False):
        for file in files:
            if fnmatch.fnmatch(file, pattern):
                file_paths.append( (os.path.join(root, file)) )
    
    return file_paths

def get_subject_image(subject_path):
    img = nib.load(subject_path) #nii file type medical image
    return img.get_fdata() #Converted to a numpy array type
    #Final conversion to PIL images type for easy saving later, also need to be in 0-255 to convert to PIL

#TODO make this return an image array and then we can bundle in another functions the slices and what not    

def save_slices(destination, image, threshold=30, rand=False):
    
    if rand:
        #Document says for randint value may be one plus the highest so I may be indexing from
        indicies = list(np.random.randint(image.shape[1], size=threshold)) 
    else:
        entropies = [ (shannon_entropy(image[:,i,:]), i) for i in range(image.shape[1]) ]
        entropies = sorted(entropies, key=lambda x: x[0], reverse=True)
        entropies = list(zip(*entropies[:threshold]))
        indicies = list(entropies[1])

    
    for index in indicies:
        temp = image[:,index,:]
        temp = Image.fromarray( (temp*255).astype(np.uint8) )
        temp.save(os.path.join(destination, str(index) + ".png"))
    
    
for experiment in ['entropy', 'random']:
    
    subject_list = os.listdir(dataset_root)
    print(subject_list)
    for subject in subject_list:
        for sess in os.listdir(os.path.join(dataset_root, subject)):
            for mode in ['mri', 'pet']:
                
                new_path = os.path.join("F:/interm/",experiment, subject, sess, mode)
                if not os.path.exists(new_path):
                    os.makedirs(new_path)
                
                #TODO need to populate the folders I made properly 
                for root, dirs, files in os.walk(os.path.join(dataset_root, subject, sess), topdown=False):
                    for file in files:
                        if fnmatch.fnmatch(file, pattern):
                           image_path = os.path.join(root, file)




# for root, dirs, files in os.walk(dataset_root):
#     for file in files:
#         if fnmatch.fnmatch(file, "*_T1w_segm-graymatter_space-Ixi549Space_modulated-on_fwhm-8mm_probability.nii.gz"):
#             print(os.path.join(root, file)) 

# img = get_subject_image("F:\caps\subjects\sub-ADNI006S1130\ses-M48\\t1\spm\dartel\group-MultiMode\sub-ADNI006S1130_ses-M48_T1w_segm-graymatter_space-Ixi549Space_modulated-on_fwhm-8mm_probability.nii.gz")
# save_slices("F:/temp", img, threshold=30, rand=True)
