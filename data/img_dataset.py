from functools import partial

# import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms as T

import numpy as np
from PIL import Image

import json
import os
from json import JSONDecoder
# from einops import rearrange

from utils import CHANNEL_TO_MODE, convert_image_to_fn



class ImageJsonDataset(Dataset):
    def __init__(
        self,
        data_dir,
        jsonl_dir,
        image_size,
        channels = 3,
        convert_image_to_mode = None,
        exts = ['.jpg', '.jpeg', '.png'],
        num_samples = None
    ):
        super().__init__()
        assert os.path.exists(data_dir), f'{str(data_dir)} must be a folder containing images'
        assert os.path.exists(jsonl_dir) and jsonl_dir.endswith('.jsonl'), f'{str(jsonl_dir)} must be a jsonl file'

        self.data_dir = data_dir  # dataset base folder dir
        self.jsonl_dir = jsonl_dir  # absolute path to the jsonl file
        self.exts = exts
        self.num_samples = num_samples
        self.image_size = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.convert_image_to_mode = convert_image_to_mode

        # parse json if it given to fetch image meta data 
        self.images_meta = self.__parse_json(self.jsonl_dir)
        assert(len(self.images_meta) != 0)

        if channels is not None and convert_image_to_mode is None:
            self.convert_image_to_mode = CHANNEL_TO_MODE.get(channels)

        self.transform = T.Compose([
            T.Lambda(partial(convert_image_to_fn, self.convert_image_to_mode)),
            T.Resize(self.image_size, antialias = True),
            T.RandomHorizontalFlip(),
            T.CenterCrop(self.image_size),
            T.ToTensor()
        ])
        # output image tensor value [0,1]
    
    def __parse_json(self, jsonl_dir):
        data = []
        assert(os.path.exists(jsonl_dir))
        with open(jsonl_dir, 'r') as f:
            for l in f:
                json_content = json.loads(l)
                img_path = os.path.join(self.data_dir,json_content['file'])
                caption = json_content['llava_caption']
                if len(json_content['llava_caption']) >0 and os.path.exists(img_path):
                    # has caption and file exist
                    data.append([img_path, caption])
                if self.num_samples and len(data) >= self.num_samples:
                    break

            f.close()
        print('read {} image samples meta info from {}'.format(str(len(data)), jsonl_dir))
        return data 

    def __len__(self):
        return len(self.images_meta)

    def __getitem__(self, index):
        img_meta = self.images_meta[index]
        img_path, img_caption = img_meta[0], img_meta[1]
        img_pil_obj = Image.open(img_path)
        img_transformed = self.transform(img_pil_obj)
        return img_transformed, img_caption
