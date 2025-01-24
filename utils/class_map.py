import os
import pickle
from tqdm import tqdm

def get_dota_class_map(ann_dir, save_path):
    '''
    Get the class map for DOTA dataset

    Args:
        ann_dir (str): Path to the directory containing DOTA annotations
        save_path (str): Path to the file to save the class
    Returns:
        dict: A dictionary mapping class names to class IDs
    '''
    class_map = {}
    class_id = 0

    if os.path.exists(save_path):
        print('[INFO] Loading class map...')
        with open(save_path, "rb") as f:
            class_map = pickle.load(f)
        print('[INFO] Class map loaded!')
        return class_map

    print('[INFO] Getting class map...')
    for file in tqdm(os.listdir(ann_dir)):
        with open(os.path.join(ann_dir, file), 'r') as f:
            for line in f:
                class_name = line.split(' ')[8]
                if class_name not in class_map:
                    class_map[class_name] = class_id
                    class_id += 1
    # save the class map to a file
    with open(save_path, 'wb') as f:
        pickle.dump(class_map, f)
    print('[INFO] Class map obtained!')
    return class_map

def expand_class_map(class_map, new_class):
    '''
    Expand the class map to include new classes

    Args:
        class_map (dict): A dictionary mapping class names to class IDs
        num_classes (str): The name of the new class
    Returns:
        dict: An expanded class map
    '''
    class_id = max(class_map.values()) + 1
    class_map[new_class] = class_id
    return class_map

if __name__ == '__main__':
    ann_dir = 'C:/Users/yangc/Developer/data/DOTA_1024_HBB/Train/annfiles'
    save_path = './cfgs/DOTA/class_map.pkl'
    class_map = get_dota_class_map(ann_dir, save_path)
    print(class_map)
