import torch
from torchvision.utils import save_image

from img_dataset import ImageJsonDataset
from multi_scale_img_dataset import ImageJsonMultiScaleDataset
import os 


img_dataset_path ={
    'SA-1B': [
        '/ML-A100/team/mm/yanghuan/data/SA-1B',
        '/ML-A100/team/mm/yanghuan/data/SA-1B/SA-1B_train.jsonl',
    ]
}

def test_single_scale_img_dataset(dataset_name: str):

    jsonl_dir = img_dataset_path[dataset_name][1]
    assert(os.path.exists(jsonl_dir))
    resolution = 256
    img_dataset = ImageJsonDataset(
        data_dir=img_dataset_path['SA-1B'][0],
        jsonl_dir=img_dataset_path['SA-1B'][1],
        image_size=resolution,
        num_samples=1000
    )

    save_root = './image_debug'
    os.makedirs(save_root, exist_ok=True)

    for i, data in enumerate(img_dataset):
        if i == 3:
            break

        print(data[0].shape)
        print(torch.max(data[0]))
        print(torch.min(data[0]))

        print(data[1])
        caption= data[1][:125]

        save_image(data[0], os.path.join(save_root, f'{resolution}_{caption}.png'), padding=0, normalize=True)
    
    return img_dataset

def test_batch_loader(dataset: torch.utils.data.Dataset):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=False,
    )

    for _, batch in enumerate(loader):
        print(batch[0].shape)
        print(len(batch[1]))


def main():
    dataset_name = 'SA-1B'
    assert dataset_name in img_dataset_path
    img_dataset = test_single_scale_img_dataset(dataset_name)

    test_batch_loader(img_dataset)

if __name__ == "__main__":
    main()