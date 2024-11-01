import os
import PIL
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST
from torchvision import datasets, transforms, models
import os
import shutil
from sklearn.model_selection import train_test_split
import re
from tqdm import tqdm, trange

np.random.seed(0)
torch.manual_seed(0)

class FungiDataset(torch.utils.data.Dataset): # inheritin from Dataset class
    def __init__(self, csv_file, root_dir="", transform=None):
        self.annotation_df = pd.read_csv(csv_file)
        self.root_dir = root_dir # root directory of images, leave "" if using the image path column in the __getitem__ method
        self.transform = transform

    def __len__(self):
        return len(self.annotation_df) # return length (numer of rows) of the dataframe

    def __getitem__(self, idx):
        image_path = self.annotation_df.iloc[idx, 1] #use image path column (index = 1) in csv file
        image = cv2.imread(image_path) # read image by cv2
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # convert from BGR to RGB for matplotlib
        class_name = self.annotation_df.iloc[idx, 2] # use class name column (index = 2) in csv file
        class_index = self.annotation_df.iloc[idx, 3] # use class index column (index = 3) in csv file
        if self.transform:
            image = self.transform(image)
        return image, class_name, class_index
    # def visualize(self, number_of_img=10, output_width=12, output_height=6):
    #     plt.figure(figsize=(output_width, output_height))
    #     for i in range(number_of_img):
    #         idx = random.randint(0, len(self.annotation_df))
    #         image, class_name, class_index = self.__getitem__(idx)
    #         ax = plt.subplot(2, 5, i+1)  # create an axis
    #         # create a name of the axis based on the img name
    #         ax.title.set_text(class_name + '-' + str(class_index))
    #         if self.transform == None:
    #             plt.imshow(image)
    #         else:
    #             plt.imshow(image.permute(1, 2, 0))
    #     plt.show()


def build_csv(directory_string, output_csv_name):
    """Builds a csv file for pytorch training from a directory of folders of images.
    Install csv module if not already installed.
    Args: 
    directory_string: string of directory path, e.g. r'.\data\train'
    output_csv_name: string of output csv file name, e.g. 'train.csv'
    Returns:
    csv file with file names, file paths, class names and class indices
    """
    import csv
    directory = directory_string
    class_lst = os.listdir(directory) #returns a LIST containing the names of the entries (folder names in this case) in the directory.
    class_lst.sort() #IMPORTANT 
    with open(output_csv_name, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['file_name', 'file_path', 'class_name', 'class_index']) #create column names
        for class_name in class_lst:
            class_path = os.path.join(directory, class_name) #concatenates various path components with exactly one directory separator (‘/’) except the last path component. 
            file_list = os.listdir(class_path) #get list of files in class folder
            for file_name in file_list:
                file_path = os.path.join(directory, class_name, file_name) #concatenate class folder dir, class name and file name
                writer.writerow([file_name, file_path, class_name, class_lst.index(class_name)]) #write the file path and class name to the csv file
    return

def clean_folder_name(name):
    # Remove all non-letter characters (A-Z, a-z)
    return re.sub(r'[^A-Za-z]', '', name)

def clean_filename(name):
    # Remove all underscores
    return name.replace('_', '')

def rename_folders_and_files(root_directory):
    for dirpath, dirnames, filenames in os.walk(root_directory, topdown=False):
        # Rename files first
        for filename in filenames:
            old_file_path = os.path.join(dirpath, filename)
            new_filename = clean_filename(filename)
            new_file_path = os.path.join(dirpath, new_filename)

            if old_file_path != new_file_path:  # Check if the name has changed
                os.rename(old_file_path, new_file_path)
                print(f"Renamed file '{old_file_path}' to '{new_file_path}'")

        # Then rename folders
        for folder_name in dirnames:
            old_folder_path = os.path.join(dirpath, folder_name)
            new_folder_name = clean_folder_name(folder_name)
            new_folder_path = os.path.join(dirpath, new_folder_name)

            if old_folder_path != new_folder_path:  # Check if the name has changed
                os.rename(old_folder_path, new_folder_path)
                print(f"Renamed folder '{old_folder_path}' to '{new_folder_path}'")
def main():
    folder = r'.\\DataSet'
    # Loading data
    # build_csv(folder, 'dataset.csv')
    
    base_dir = folder  # path to your dataset
    # train_dir = os.path.join(base_dir, 'train')
    # test_dir = os.path.join(base_dir, 'test')
    val_dir = os.path.join(base_dir, 'prediction')

    test_size = 0.1  # 20% for test
    val_size = 0.1   # 20% of remaining for validation
    # rename_folders_and_files(train_dir)
    # rename_folders_and_files(test_dir)
    rename_folders_and_files(val_dir)
    
    # Create directories for train, test, and validation splits
    # for split_dir in [train_dir, test_dir, val_dir]:
    #     os.makedirs(split_dir, exist_ok=True)

    # DELETE
    # Pesisa-ostracoderma
    #
    # Loop through each class in the dataset
    # for class_name in os.listdir(base_dir):
    #     if class_name == "test" or class_name == "train" or class_name == "validation":
    #         continue
    #     print("Doing: " + class_name)
    #     class_path = os.path.join(base_dir, class_name)
    #     if os.path.isdir(class_path):
    #         # Get a list of files in the class folder
    #         files = os.listdir(class_path)
    #         files = [f for f in files if os.path.isfile(os.path.join(class_path, f))]
            
    #         # Split files into train and test
    #         train_files, test_files = train_test_split(files, test_size=test_size, random_state=42)
            
    #         # Further split train into train and validation
    #         train_files, val_files = train_test_split(train_files, test_size=val_size, random_state=42)
            
    #         # Create class directories within train, test, and validation
    #         for split, file_list in zip([train_dir, test_dir, val_dir], [train_files, test_files, val_files]):
    #             split_class_dir = os.path.join(split, class_name)
    #             os.makedirs(split_class_dir, exist_ok=True)
                
    #             # Move files to the respective directories
    #             for file_name in file_list:
    #                 src = os.path.join(class_path, file_name)
    #                 dest = os.path.join(split_class_dir, file_name)
    #                 shutil.copyfile(src, dest)
                    
    

    print("Dataset split into train, test, and validation folders successfully.")
if __name__ == "__main__":
    main()