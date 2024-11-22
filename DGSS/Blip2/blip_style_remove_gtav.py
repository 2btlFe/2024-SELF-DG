import argparse
import torch
import os

import yaml
from PIL import Image
import json

# from gpt_interface import GPTInterface
from captioning.blip import BLIPInterface

from captioning.gpt_interface import GPTInterface
# from datasets.datamodules import PascalVOCDataModule
import ipdb
from tqdm import tqdm

class BlipDataset(torch.utils.data.Dataset):

    def __init__(self, img_paths):
        self.img_paths = img_paths

    def __len__(self):
        return len(self.img_paths)

    def get_image(self, idx, return_img_file=False):
        img_file = self.img_paths[idx]

        image = Image.open(img_file).convert("RGB")

        ret_list = [image]

        if return_img_file:
            ret_list.append(img_file)

        return ret_list


def get_files_recursively(directory):
    all_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


def parse_args():
    parser = argparse.ArgumentParser(description='Caption a dataset using BLIP2')
    parser.add_argument('--dataset_path', help='dataset path')
    parser.add_argument('--dataset', help='dataset to caption')
    parser.add_argument('--max_new_tokens', type=int, default=77, help='max number of tokens generated')
    parser.add_argument('--min_new_tokens', type=int, default=0, help='min number of tokens generated')
    parser.add_argument('--params_name', type=str, default=None, help='params string (added to end of file name)')
    parser.add_argument('--profile', action='store_true', default=False)
    parser.add_argument('--use_gpt_for_style_removal', action='store_true', default=False)
    parser.add_argument('--overwrite', action='store_true', default=False)

    # Blip Generation parameters
    return parser.parse_args()


def main():
    args = parse_args()

    # parse blip_generation_parameters
    blip_generation_dict = {
        'max_new_tokens': args.max_new_tokens,
        'min_new_tokens': args.min_new_tokens,
    }

    img_name_dict = None
    train_path = args.dataset_path
    train_files = os.listdir(train_path)
    train_paths = [os.path.join(train_path, f) for f in train_files]
    img_paths = train_paths
    
    img_paths = []
    for (path, dir, files) in os.walk(train_path):
        for filename in files:
            ext = os.path.splitext(filename)[-1]
            if ext == '.png' or ext == '.jpg':
                img_paths.append(os.path.join(path, filename))   

    # labels
    labels = [path.split('_')[0] for path in img_paths]
    
    dataset = BlipDataset(img_paths)

    bi = BLIPInterface(dataset, args.dataset, args.params_name, blip_generation_dict)
    # bi(batch_size=32)  # set to whatever works for your GPU

    # bi = BLIPInterface(dataset, args.dataset, args.params_name, blip_generation_dict)
    captions = bi(overwrite=args.overwrite, img_name_dict=img_name_dict, batch_size=128)

    if args.use_gpt_for_style_removal:
        cfg = yaml.load(open("./gpt_cfg.yaml", "r"), Loader=yaml.FullLoader)
        gpt = GPTInterface(cfg, classnames=['road', 'sidewalk', 'building', 'wall', 'fence', 'pole',
                'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky',
                'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle',
                'bicycle'])
        all_captions = []
        all_captions_keys = []
        for key, value in captions.items():
            all_captions.extend(value['captions'])
            all_captions_keys.extend([key])
        # prompt = "remove any descriptions of the style of the image, just leaving its content/objects, remove any mention of image; depiction; portrayal; painting; watercolor; comic; cartoon etc. for the following captions: \n "
        prompt = "remove any descriptions of the style of the image, just leaving its content/objects, remove any mention of domain information like gta 5; gta; grand theft auto; gtav etc, remove any mention of image; depiction; portrayal; painting; watercolor; comic; cartoon etc. for the following captions: \n "
        
        # prompt = "remove any descriptions of the style of the image, just leaving its content/objects, remove any mention of domain information like gta 5; gta; grand theft auto; gtav etc, must include {}, remove any mention of image; depiction; portrayal; painting; watercolor; comic; cartoon etc. for the following captions: \n "
        # constraint_prompt = "Provide each caption as a single sentence without any list formatting, bullets, or symbols, beginning directly with the content of the caption"

        splits = 1500   #3000
        num_captions = len(all_captions)
        gpt_batch_size = num_captions // splits
        outs = []

        print(f"number of splits : {splits}")

        for i in tqdm(range(splits)):
            gpt_batch = all_captions[i * gpt_batch_size:(i + 1) * gpt_batch_size]
            label_batch = labels[i * gpt_batch_size:(i + 1) * gpt_batch_size]
            print(len(gpt_batch))
            print(f"gpt_batch - {gpt_batch}")

            print(len(label_batch))
            print(f"label_batch - {label_batch}")

            assert len(gpt_batch) == len(label_batch)
            
            # prompt = prompt.format(label_batch)
            gpt_batch = '\n '.join(gpt_batch)
            _prompt = prompt + gpt_batch #+ constraint_prompt
            check = False
            # retry until we get the right number of captions --- sometimes GPT fails
            while not check:
                out = gpt.general_gpt_task(_prompt)
                cleaned_captions = out.split('\n')
                if len(cleaned_captions) == len(gpt_batch.split('\n')):
                    check = True
                    print('done')
                else:
                    print('retrying')
            print(len(cleaned_captions))
            outs.append(cleaned_captions)
        # split captions by newline
        bla = [item for sublist in outs for item in sublist]
        cleaned_captions = bla
        new_captions = {}

        for key, cap in zip(all_captions_keys, cleaned_captions):
            # new_captions[key] = {'captions': [cap], 'class_label': captions[key]['class_label']}    # class_label
            new_captions[key] = {'captions': [cap]}    # class_label
        captions = new_captions
        out_path = "blip_captions_gtav/"
        os.makedirs(out_path, exist_ok=True)
        os.path.join(out_path, args.dataset + '_captions.json')
        with open(os.path.join(out_path, args.dataset + '_captions.json'), 'w') as f:
            # json.dump(image_file_captions, f, indent=2)
            json.dump(captions, f)

if __name__ == '__main__':
    main()
