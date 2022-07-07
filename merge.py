import os
import json
import tqdm

INPUT_ROOT = './data/2D_data'
OUTPUT_ROOT = './data/2D_data'

def main():
    data_2d_names = os.listdir(INPUT_ROOT)
    data_2d_names = [
        name for name in data_2d_names if not name.startswith('merge')]
    print(data_2d_names)
    if len(data_2d_names) == 0:
        print('No data to process')
        return

    output_json = []

    for file_name in tqdm.tqdm(data_2d_names):
        json_open_2d = open(os.path.join(INPUT_ROOT, file_name), 'r', encoding='utf-8')
        
        try:        
            json_load_2d = json.load(json_open_2d)
        except UnicodeDecodeError:
            print('UnicodeDecodeError')
            continue

        output_json += json_load_2d

    f = open(os.path.join(OUTPUT_ROOT, f'merge_cat_train.json'), 'w')
    json.dump(output_json, f)


if __name__ == '__main__':
    main()