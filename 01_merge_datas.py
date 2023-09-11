#merge images from different sources to one json file, removing duplicates and discrepancies

import os

from PIL import Image
from tqdm import tqdm

import myutils.file_utils as file


image_dir = '01_merged_datas/images/'
target_dir = '01_merged_datas/images_resized/'


def process_image(input_path, output_path, target_size=(512, 512), target_ext = 'JPEG'):
    try:
        with Image.open(input_path) as img:
            img = img.convert("RGB")
            img.thumbnail(target_size)
            img.save(output_path, "JPEG")
    except Exception as e:
        print(f"Error processing {input_path}: {e}")

if not os.path.exists(target_dir):
    os.makedirs(target_dir)
    
all_file_names = file.get_file_names_in_dir(image_dir)
for file_name in tqdm(all_file_names):
    process_image(image_dir + file_name, target_dir + f"{os.path.splitext(file_name)[0]}.jpg")



'''
import pandas as pd

import myutils.file_utils as file

image_dir = '01_merged_datas/images/'
output_csv_path = '01_merged_datas/image_names.csv'

all_file_names = file.get_file_names_in_dir(image_dir)

file_names_df = pd.DataFrame(all_file_names)
file_names_df.to_csv(output_csv_path, header = ['name'], index = False)
'''