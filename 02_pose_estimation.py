'''
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
pct/env2/Scripts/activate
cd pct
python ../02_pose_estimation.py
'''

from tqdm import tqdm
import pandas as pd
from multiprocessing import Process

from pct.vis_tools.run_keypoints import inference

import myutils.file_utils as file


image_dir = '../01_merged_datas/images_resized/'
#output_csv_path = '../01_merged_datas/image_datas.csv'

slice_size = 100


all_file_names = file.get_file_names_in_dir(image_dir)
slice_idxs = [i for i in range(0, len(all_file_names), slice_size)]
slice_idxs[-1] = len(all_file_names)


for i in range(88,len(slice_idxs)):
    output_csv_path = f'../01_merged_datas/image_datas{i}.csv'
    
    print(f"Slice {i} START!")
    pose_results = inference(all_file_names[slice_idxs[i] : slice_idxs[i+1]], image_dir)

    print(f"Slice {i} DONE! ({len(pose_results)})")

    pose_results_df = pd.DataFrame(pose_results)
    pose_results_df.to_csv(output_csv_path, header = ['name', 'person_count', 'bbox', 'keypoints',])

    print(f"Saved data as {output_csv_path}")


