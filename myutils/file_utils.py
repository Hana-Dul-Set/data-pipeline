import json
import os
from PIL import Image

def get_image_size(image_path):
    with Image.open(image_path) as image:
        width, height = image.size
    return width, height

def read_json(file_path):
    print(f"Reading {file_path}...", end = '')
    with open(file_path, 'r') as json_file:
        data = json.load(json_file)
    print("Done!")
    return data

def save_as_json(data, file_path):
    print(f"Saving {file_path}...", end = '')
    with open(file_path, 'w') as json_file:
        json.dump(data, json_file)
    print("Done!")

def get_file_names_in_dir(dir_path):
    file_names = os.listdir(dir_path)
    return [file_name for file_name in file_names if os.path.isfile(os.path.join(dir_path, file_name))]

def get_image_names_recursive(dir_path):
    image_extensions = ['jpg', 'jpeg', 'png', 'webp', 'heic']  # Add or remove extensions as needed
    image_file_names = []

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.lower().endswith(tuple(image_extensions)):
                image_file_names.append(os.path.join(root, file))

    return image_file_names