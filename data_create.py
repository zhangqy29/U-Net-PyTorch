import numpy as np
import nibabel as nib
import os.path


# source folder where the .nii files are located 
source_folder = '/home/zhangqy/CT/data/LITS17/'
# destination folder where the subfolders with npy files will go
destination_folder = '/home/zhangqy/CT/data/LiTS17/Test/'



if not os.path.exists(destination_folder):
	os.makedirs(destination_folder)


for file_name in os.listdir(source_folder):

    # create new file name by stripping .nii.gz and adding .npy
    new_file_name = file_name[:-4]
    num=int(new_file_name.split("-")[-1])
    if num in list(range(101)):
        continue

    # load file
    data = nib.load(os.path.join(source_folder, file_name))

    # convert to numpy
    data = data.get_data()

    
    # transpose so the z-axis (slices) are the first dimension
    data = np.transpose(data, (2, 0, 1))

    # loop through the slices
    for i, z_slice in enumerate(data):

        # save at new location (train or val)
        np.save(os.path.join(destination_folder, new_file_name + '_' + str(i)), z_slice)
