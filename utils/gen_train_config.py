# Description: Generate training configuration file for training the model.

import os
import yaml
import pickle

from convert import get_dota_class_map

def gen_train_config(train_root, save_path, map_path):
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
    class_map = get_dota_class_map(ann_dir, map_path)

    # Generate the training configuration file
    # Swap keys and values in class_map
    class_map_swapped = {v: k for k, v in class_map.items()}
    
    config = {
        'path': str(os.path.dirname(train_root)),  # Get parent directory of train_root
        'train': '/train/',
        'val': '/val/',

        'names': class_map_swapped,
    }

    # Reorder the dictionary to put 'names' at the end
    ordered_config = {k: config[k] for k in ['path', 'train', 'val']}
    ordered_config['names'] = config['names']

    with open(save_path, 'w') as f:
        yaml.dump(ordered_config, f)



if __name__ == '__main__':
    train_root = 'C:/Users/yangc/Developer/data/dota_small_1024/train'
    save_path = './cfgs/dota_small/train_config.yaml'
    map_path = './cfgs/dota_small/class_map.pkl'
    gen_train_config(train_root, save_path, map_path)