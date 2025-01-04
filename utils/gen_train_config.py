# Description: Generate training configuration file for training the model.

import os
import yaml
import pickle

from convert import get_dota_class_map

def gen_train_config(train_root, save_path):
    '''
    Generate training configuration file for training the model.

    Args:
        data_path (str): Path to the data root
        save_path (str): Path to save the training configuration file

    e.g.
    data_path = 'PATH/TO/TRAIN/DIR'
    save_path = 'PATH/TO/SAVE/TRAIN/CONFIG'
    '''
    # Get the class map
    ann_dir = os.path.join(train_root, 'annfiles')
    class_map = get_dota_class_map(ann_dir, save_path='./configs/DOTA/class_map.pkl')

    # Generate the training configuration file
    # Swap keys and values in class_map
    class_map_swapped = {v: k for k, v in class_map.items()}
    
    config = {
        'path': str(os.path.dirname(train_root)),  # Get parent directory of train_root
        'train': '/Train/',
        'val': '/Val/',

        'names': class_map_swapped,
    }

    # Reorder the dictionary to put 'names' at the end
    ordered_config = {k: config[k] for k in ['path', 'train', 'val']}
    ordered_config['names'] = config['names']

    with open(save_path, 'w') as f:
        yaml.dump(ordered_config, f)



if __name__ == '__main__':
    train_root = 'C:/Users/yangc/Developer/data/DOTA_1024_HBB/train'
    save_path = './configs/DOTA/train_config.yaml'
    gen_train_config(train_root, save_path)