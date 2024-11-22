import os

def find_folders_with_multiple_subfolders(root_folder):
    folders_with_multiple_subfolders = []
    
    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]
            if len(subfolders) > 1:
                folders_with_multiple_subfolders.append(folder_path)
    
    return folders_with_multiple_subfolders

# Example usage
root_folder = r'E:\fredd\Uni\Thesis\Image-Processing\Data\AllData'
folders = find_folders_with_multiple_subfolders(root_folder)
print(f"Folders with more than one subfolder: {folders}")