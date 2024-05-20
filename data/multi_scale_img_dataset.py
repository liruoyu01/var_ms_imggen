import os
import json
from functools import partial
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

import torch
from torchvision import transforms as T
from torch.utils.data import Dataset

from utils import CHANNEL_TO_MODE, get_closest_ratio, convert_image_to_fn


class ImageJsonMultiScaleDataset(Dataset):
    def __init__(
        self,
        data_dir,
        jsonl_dir,
        channels = 3,
        convert_image_to_mode = None,
        num_samples = None,
        max_length=120,
        save_cache=True,
        shard_size=5000,
        max_retry=10,
        torch_dtype=torch.float32,
        **kwargs
    ):
        super().__init__()
        assert os.path.exists(data_dir), f'{str(data_dir)} must be a folder containing images'
        assert os.path.exists(jsonl_dir) and jsonl_dir.endswith('.jsonl'), f'{str(jsonl_dir)} must be a jsonl file'

        self.root = data_dir
        self.jsonl_dir = jsonl_dir  # absolute path to the jsonl file

        self.torch_dtype = torch_dtype

        self.cache_dir = os.path.join(self.root, 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        self.save_cache = save_cache
        self.shard_size = shard_size
        
        self.num_samples = num_samples
        self.convert_image_to_mode = convert_image_to_mode

        self.base_size = int(kwargs['aspect_ratio_type'].split('_')[-1])
        self.max_length = max_length
        self.aspect_ratios = eval(kwargs.pop('aspect_ratio_type'))       # base aspect ratio
        self.aspect_ratio_index = defaultdict(list)
        self.aspect_ratio_cnt = defaultdict(int)
        self.full_img_info = []
        self.max_retry = max_retry

        self.images_meta = []
        self.cache_pth_dir_set = set()
        self.__parse_json_save_meta(self.jsonl_dir)
        assert(len(self.images_meta) != 0)

        if channels is not None and convert_image_to_mode is None:
            self.convert_image_to_mode = CHANNEL_TO_MODE.get(channels)
    
    def get_aspect_ratio_index(self):
        return self.aspect_ratio_index

    def get_aspect_ratio_cnt(self):
        return self.aspect_ratio_cnt
    
    def __parse_json_save_meta(self, jsonl_dir):

        with open(jsonl_dir, 'r') as f:
            # read all json lines
            json_lines = list(map(json.loads, f))
            # truncate first num_samples of json_lines as lines_to_load
            if self.num_samples and self.num_samples <= len(json_lines):
                lines_to_load = json_lines[:self.num_samples]
            else:
                lines_to_load = json_lines

            self.total_data_count = len(lines_to_load)

            # read and parse lines_to_load line by line
            for idx in tqdm(range(len(lines_to_load))):
                json_content = lines_to_load[idx]
                img_path = os.path.join(self.root,json_content['file'])
                assert(os.path.exists(img_path))
                caption_text = json_content['llava_caption']
                if len(caption_text)>0 and os.path.exists(img_path):
                    # has caption and file exist
                    img_pil_obj = Image.open(img_path)
                    org_h, org_w = img_pil_obj.size
                    # calc closest_size and closest_ratio, resize_size
                    closest_size, closest_ratio = get_closest_ratio(org_h, org_w, self.aspect_ratios)
                    
                    closest_size = list(map(lambda x: int(x), closest_size))
                    if closest_size[0] / org_h > closest_size[1] / org_w:
                        resize_size = closest_size[0], int(org_w * closest_size[0] / org_h)
                    else:
                        resize_size = int(org_h * closest_size[1] / org_w), closest_size[1]
                        
                    self.images_meta.append(
                        {
                            'img_path': img_path, 
                            'caption_text':caption_text, 
                            'idx': idx,
                            'resize_size': resize_size,
                            'closest_size': closest_size,
                            'img_height': org_h,
                            'img_width': org_w,
                            'aspect_ratio': closest_ratio,
                        }
                    )
                    self.aspect_ratio_index[closest_ratio].append(idx)
                    self.aspect_ratio_cnt[closest_ratio] += 1
                
                # save when processed images_meta reach shard_size
                if idx > 0 and len(self.images_meta) % self.shard_size == 0 and self.save_cache:
                    cache_file = 'full_img_info_shard{}_{}_npz_file.pth'.format(str(self.shard_size), str(idx//self.shard_size))
                    cache_pth_dir = os.path.join(self.cache_dir, cache_file)
                    torch.save(self.images_meta, cache_pth_dir)
                    if cache_pth_dir not in self.cache_pth_dir_set: self.cache_pth_dir_set.add(cache_pth_dir)
                    self.images_meta = []
            
            if len(self.images_meta) > 0: 
                cache_file = 'full_img_info_shard{}_{}_npz_file.pth'.format(str(self.shard_size), str(idx//self.shard_size))
                cache_pth_dir = os.path.join(self.cache_dir, cache_file)
                torch.save(self.images_meta, cache_pth_dir)
                if cache_pth_dir not in self.cache_pth_dir_set: self.cache_pth_dir_set.add(cache_pth_dir)

            f.close()
        
        print('read {} line of jsonl metadata from {}, and saved {} cache files'.format(str(len(lines_to_load)), jsonl_dir, len(self.cache_pth_dir_set)))
        print('we have {} resized image resolutions'.format(len(self.aspect_ratio_index.keys())))

    def __len__(self):
        return self.total_data_count

    def __getitem__(self, index):
        
        cache_file = 'full_img_info_shard{}_{}_npz_file.pth'.format(str(self.shard_size), str(index//self.shard_size))
        cache_pth_dir = os.path.join(self.cache_dir, cache_file)
        try:
            cacheed_img_info = torch.load(cache_pth_dir)  # list of dict , each dict is img meta for one image
            img_info_for_idx = cacheed_img_info[int(index % self.shard_size)]
        except:
            raise Exception('the data of index {} NOT found in processed cache file at {}!'.format(str(index), cache_pth_dir))
        
        assert(index == img_info_for_idx['idx'])
        
        # get img transformed tensor
        img_pil_obj = Image.open(img_info_for_idx['img_path'])
        transform_fn = T.Compose([
            T.Lambda(partial(convert_image_to_fn, self.convert_image_to_mode)),
            T.Resize(img_info_for_idx['resize_size'], antialias = True),
            T.RandomHorizontalFlip(),
            T.CenterCrop(img_info_for_idx['closest_size']),
            T.ToTensor()
        ])
        img_transformed = transform_fn(img_pil_obj)
        img_transformed_detach = img_transformed.detach()
                
        data_info = {
            'img_height': img_info_for_idx['img_height'],
            'img_width': img_info_for_idx['img_width'],
            'img_hw': [img_info_for_idx['img_height'], img_info_for_idx['img_width']],
            'aspect_ratio': img_info_for_idx['aspect_ratio'],
            'caption': str(img_info_for_idx['caption_text'])
        }
        
        return img_transformed_detach, data_info

