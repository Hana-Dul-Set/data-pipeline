#merge images from different sources to one json file, removing duplicates and discrepancies

import pandas as pd

import myutils.file_utils as file

image_dir = '01_merged_datas/images/'
output_csv_path = '01_merged_datas/image_names.csv'


all_file_names = file.get_file_names_in_dir(image_dir)

file_names_df = pd.DataFrame(all_file_names)
file_names_df.to_csv(output_csv_path, header = ['name'], index = False)