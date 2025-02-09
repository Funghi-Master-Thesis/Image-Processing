import os
import pandas as pd
import csv
from scipy import ndimage as ndi
from skimage.morphology import disk
from skimage.segmentation import watershed
from skimage.filters import rank
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
import numpy as np
import cv2
from collections import Counter



def build_annotation_dataframe(image_location, annot_location, output_csv_name):
    """Builds dataframe and csv file for pytorch training from a directory of folders of images.
    Install csv module if not already installed.
    Args: 
    image_location: image directory path, e.g. r'.\data\train'
    annot_location: annotation directory path
    output_csv_name: string of output csv file name, e.g. 'train.csv'
    Returns:
    csv file with file names, file paths, class names and class indices
    """
    class_lst = os.listdir(
        image_location)  # returns a LIST containing the names of the entries (folder names in this case) in the directory.
    class_lst.sort()  # IMPORTANT
    with open(os.path.join(annot_location, output_csv_name), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['file_name', 'file_path', 'class_name',
                        'class_index'])  # create column names
        for class_name in class_lst:
            if class_name in ['ColorFeature.csv', 'Colors.csv', 'all.csv', 'AllColors.csv']:
                continue
            # concatenates various path components with exactly one directory separator (‘/’) except the last path component.
            class_path = os.path.join(image_location, class_name)
            # get list of files in class folder
            file_list = os.listdir(class_path)
            for file_name in file_list:
                # concatenate class folder dir, class name and file name
                file_path = os.path.join(image_location, class_name, file_name)
                # write the file path and class name to the csv file
                writer.writerow(
                    [file_name, file_path, class_name, class_lst.index(class_name)])
    return pd.read_csv(os.path.join(annot_location, output_csv_name))


def split_color():
    image_location = 'E:/fredd/Uni/Thesis/Datasets/AllDatasets/DataSetCutLast2Days/'
    # Load the original CSV file
    input_file = image_location + "ColorFeature.csv"  # Replace with your input file path
    output_file = image_location + "Colors.csv"  # Replace with your output file path

    # Read the CSV file
    df = pd.read_csv(input_file)

    # Remove the specified column (e.g., 'ColumnName')
    columns_to_remove = ["file_path", "class_name", "class_index"]  # Replace with column names to remove

    # Remove the specified columns
    df.drop(columns=columns_to_remove, inplace=True, errors='ignore')

    # Save the modified DataFrame to a new CSV file
    df.to_csv(output_file, index=False)

def create_csv():
    data = 'E:/fredd/Uni/Thesis/Datasets/AllDatasets/DataSetCutLast2Days/'
    
    df = build_annotation_dataframe(image_location=data, annot_location='E:/fredd/Uni/Thesis/Datasets/AllDatasets/DataSetCutLast2Days/', output_csv_name='all.csv')

def combine_color():
    data = 'E:/fredd/Uni/Thesis/Datasets/AllDatasets/DataSetCutLast2Days/'
    
    file1 = data + "all.csv"  # Replace with the path to the first CSV file
    file2 = data + "Colors.csv"  # Replace with the path to the second CSV file
    output_file = data + "AllColors.csv"  # Replace with the output file path

    # Read the CSV files
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)
    df3 = pd.read_csv(data + "ColorFeature.csv")
    
    

    # Common column to merge on
    common_column = "file_name"  # Replace with the name of the common column

    # Merge the two DataFrames
    merged_df = pd.merge(df1, df2, on=common_column, how='inner')  # 'inner' merge

    # Save the merged DataFrame to a new CSV file
    merged_df.to_csv(output_file, index=False)
    diff_rows = pd.concat([merged_df, df3]).drop_duplicates(keep=False)
    print("Differences in rows:")
    print(diff_rows)
    
    if set(merged_df.columns) != set(df3.columns):
        print("The column sets are different.")
        print("Columns in file1 but not in file2:", set(df1.columns) - set(df2.columns))
        print("Columns in file2 but not in file1:", set(df2.columns) - set(df1.columns))

def compare_linux_windows():
    data = 'E:/fredd/Uni/Thesis/Datasets/AllDatasets/DataSetCutLast2Days/'
    


    # Read the CSV files

    df1 = pd.read_csv(data + "AllColors.csv")
    df2 = pd.read_csv(data + "AllColorsLinux.csv")
    diff_rows = pd.concat([df1, df2]).drop_duplicates(keep=False)
    print("Differences in rows:")
    print(diff_rows)
    if set(df1.columns) != set(df2.columns):
        print("The column sets are different.")
        print("Columns in file1 but not in file2:", set(df1.columns) - set(df2.columns))
        print("Columns in file2 but not in file1:", set(df2.columns) - set(df1.columns))

def main():
    compare_linux_windows()

if __name__ == "__main__":
    main()