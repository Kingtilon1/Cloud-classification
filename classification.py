from torchvision.datasets import ImageFolder
from torchvision import transforms
import pandas as pd
import shutil
import os

## Lets store the csv into a data fram to put the images in the correct directory based on their label
train=pd.read_csv('train.csv')

## dictionary to map label to name
cloud_type = {
    0: 'cirriform',
    1: 'clear_sky',
    2: 'cumulonimbus',
    3: 'cumulus',
    4: 'high_cumuliform',
    5: 'stratiform',
    6: 'stratocumulus'
}
# Base path setup 
base_path = os.path.dirname(os.path.abspath(__file__))

## path setup for images/train
source_path = os.path.join(base_path, 'images', 'train')

target_base = os.path.join(base_path, 'cloud_train') 
## if the label is in the dictionary, move the subsequent jpg number into the correct folder that matches the cloud tpye
for index, row in train.iterrows():
     type_of_cloud = row['label']
     ### stores the file name 
     image_id = row['id']
     if type_of_cloud in cloud_type:
          image_to_move = os.path.join(source_path, image_id)
          target_dir=os.path.join(target_base, cloud_type[type_of_cloud])
          place_to_move = os.path.join(target_dir, image_id)
          shutil.move(image_to_move, place_to_move)
          print(f"Moved {image_id} to {target_dir}")
     else:
        print(f"Warning: {image_id} does not exist in {source_path} and cannot be moved.")