import os
import cv2
import numpy as np

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
data_folder_path = 'E:\\fredd\\Uni\\Thesis\\Image-Processing\\Data\\AllData'
output_folder = os.path.join(base_path, 'Data', 'Output', 'DataSinglePlate')

finished_ibt = 'Preprocessing\\filters\\excludeclean.txt'


lines = open(finished_ibt).read().splitlines()

def apply_circular_mask(image, scale_factor=0.675):
    """Apply a circular mask to the image with an adjustable scale factor."""
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    center = (width // 2, height // 2)
    radius = int(min(center[0], center[1], width - center[0], height - center[1]) * scale_factor)
    cv2.circle(mask, center, radius, (255, 255, 255), -1)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    return masked_image

def cut_to_boundingbox(image):
    """Cut the image to the bounding box of the circle."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    x, y, w, h = cv2.boundingRect(thresh)
    image = image[y:y+h, x:x+w]
    return image

def extract_and_save_petri_dishes(image_path, output_folder, resize_dim=None, ibt_number=''):
    """Extract each petri dish from the image and save them with labels."""
    # print(f"Processing image: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load image: {image_path}")
        return

    height, width, _ = image.shape
    # print(f"Image dimensions: {width}x{height}")

    rows, cols = 3, 2
    petri_height = height // rows
    petri_width = width // cols

    for row in range(rows):
        for col in range(cols):
            x_start = col * petri_width
            y_start = row * petri_height
            x_end = x_start + petri_width
            y_end = y_start + petri_height

            petri_dish = image[y_start:y_end, x_start:x_end]
            # petri_dish = apply_circular_mask(petri_dish)
            # petri_dish = cut_to_boundingbox(petri_dish)
            # Resize if resize_dim is provided
           
            #If the array is empty continue to the next iteration
            if petri_dish.size == 0:
                continue
            # petri_dish = cv2.resize(petri_dish, resize_dim)


            label = f"{os.path.splitext(os.path.basename(image_path))[0]}_row_{row+1}_col_{col+1}.jpg"
            output_path = os.path.join(output_folder, label)
            success = cv2.imwrite(output_path, petri_dish)
            if not success:
                print(f"Failed to save cropped image: {output_path}")

def main():
    os.makedirs(output_folder, exist_ok=True)
    print(f"Output folder created: {output_folder}")
    print(f"Output folder path: {output_folder}")

    # Process each image in the data folder
    print(f"Processing images in folder: {data_folder_path}")
    for fungi in os.listdir(data_folder_path):
        folder_path = os.path.join(data_folder_path, fungi)
    
        correct_folder = os.listdir(folder_path)[0]
        if correct_folder != "RIS1_0_TL_20_preset":
            print("Weird Folder, skipping" + correct_folder)
            continue
        folder_path = os.path.join(folder_path, os.listdir(folder_path)[0])
        if fungi in lines:
            print("Already processed " + fungi + ", skipping!")
            continue
        fungi_path = os.path.join(output_folder, fungi)
        os.makedirs(fungi_path, exist_ok=True)
        print(f"Found fungi: {fungi_path}")
        for image_file in os.listdir(folder_path):
            if image_file.lower().endswith('.jpg') or image_file.lower().endswith('.jpeg'):
                image_path = os.path.join(folder_path, image_file)
                print(f"Processing file: {image_path}")
                extract_and_save_petri_dishes(image_path, fungi_path, resize_dim=(244, 244))  # Adjust resize_dim as needed
        with open(finished_ibt, 'r') as file:
            content = file.read()
        ibts = content + fungi + "\n"
        with open(finished_ibt, 'w') as file:
            file.write(ibts)
    print("Processing complete.")

# Using the special variable 
# __name__
if __name__=="__main__":
    main()
# Ensure the output folder exists
