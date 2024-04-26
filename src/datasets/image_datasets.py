import math
import random

from PIL import Image
import numpy as np
from torch.utils.data import DataLoader, Dataset
from io import BytesIO
import torch as th
import torch
import torch.nn.functional as F

import pickle
from torchvision import transforms
import os
import json
from einops import rearrange


def collate_fn(examples):
    batch = {}
    origin_image = [example["origin_image"] for example in examples]
    image = [example["image"] for example in examples]
    normal = albedo = rendered = None
    hint_list = []
    if 'normal' in examples[0]:
        normal = [example["normal"] for example in examples]
        normal = th.stack(normal, 0)
        hint_list.append(normal)
    if 'albedo' in examples[0]:
        albedo = [example["albedo"] for example in examples]
        albedo = th.stack(albedo, 0)
        hint_list.append(albedo)
    if 'rendered' in examples[0]:
        rendered = [example["rendered"] for example in examples]
        rendered = th.stack(rendered, 0)
        hint_list.append(rendered)

    prompt = [example["prompt"] for example in examples]
    id_feature = [example["id_feature"] for example in examples]

    if "mask" in examples[0]:
        mask = [th.tensor(example["mask"]) for example in examples]
        mask = th.stack(mask, 0)
        batch["mask"] = mask

    if "full_mask" in examples[0]:
        full_mask = [th.tensor(example["full_mask"]) for example in examples]
        full_mask = th.stack(full_mask, 0)
        batch["full_mask"] = full_mask

    image = th.stack(image, 0)
    image = rearrange(image, 'b c h w -> b h w c')

    id_feature = th.stack(id_feature, 0)
    id_feature = id_feature.unsqueeze(dim=1)

    hint = th.cat(hint_list, dim=1)
    hint = rearrange(hint, 'b c h w -> b h w c')

    image = image.to(memory_format=th.contiguous_format).float()
    hint = hint.to(memory_format=th.contiguous_format).float()

    batch.update({
        "jpg": image,
        "cond_img": origin_image,
        "hint": hint,
        "txt": prompt,
        "id": id_feature,
    })

    return batch


class ImageDataset(Dataset):
    def __init__(
        self,
        path,
        is_clip_mask=0,
        use_mask=0,
        id_file='',
        text_file='',
        drop_text=0,
        cond=None,
        **kwargs
    ):
        super().__init__()

        self.path = path
        self.is_clip_mask = is_clip_mask
        self.use_mask = use_mask
        self.id_file = id_file
        self.text_file = text_file
        self.drop_text = drop_text
        self.cond = cond

        transform = [
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]

        self.transform = transforms.Compose(transform)

        with open(os.path.join(self.path, 'imgid.txt'), 'r') as f:
            self.idxs = f.read().split()

        print(f'#Total data: {len(self.idxs)}')

    def open_text(self):

        with open(os.path.join(self.path, self.text_file), 'r') as f:
            self.text = json.load(f)

    def check_text(self):
        if not hasattr(self, 'text'):
            self.open_text()

    def __len__(self):
        return len(self.idxs)

    def _read_data(self, index, clip_index):

        # index: imagename
        normal = Image.open(os.path.join(self.path, 'deca_data', 'normal', f'{index}.png'))
        albedo = Image.open(os.path.join(self.path, 'deca_data', 'albedo', f'{index}.png'))
        rendered = Image.open(os.path.join(self.path, 'deca_data', 'rendered', f'{index}.png'))
        image = Image.open(os.path.join(self.path, 'Full_align_sup', f'{index}.png'))
        clip_image = Image.open(os.path.join(self.path, 'Face_align_sup', f'{clip_index}.png'))
        normal = self.transform(normal)
        albedo = self.transform(albedo)
        rendered = self.transform(rendered)

        return image, clip_image, normal, albedo, rendered

    def get_id_feature(self, index, path=None, id_dir='id_features'):
        if path is None:
            path = self.path
        id_feature = torch.load(os.path.join(path, id_dir, f'{index}.pt'), map_location='cpu')
        return id_feature

    def __getitem__(self, index):

        imgid = self.idxs[index]

        data_dict = self.get_data(index)

        return data_dict

    def get_data(self, index):

        data_dict = {}

        self.check_text()
        index = self.idxs[index]

        clip_index = index

        image, origin_image, normal, albedo, rendered = self._read_data(index, clip_index)
        # origin_image is clip_image

        image = self.transform(image)

        # get text
        if index in self.text:
            prompt = self.text[index]
        elif index+'.png' in self.text:
            prompt = self.text[index+'.png']

        if self.drop_text != 0:
            drop_or_not = np.random.random()
            if drop_or_not < self.drop_text:
                prompt = ""

        id_feature = self.get_id_feature(clip_index)

        if self.use_mask != 0:
            full_mask = np.load(os.path.join(self.path, 'Full_align_sup_mask', 'mask', f'{index}.npy'))
            data_dict["full_mask"] = full_mask

        if self.is_clip_mask != 0:
            image_wobg = np.asarray(origin_image).copy()
            mask = np.load(os.path.join(self.path, 'Face_align_sup_mask', 'mask', f'{clip_index}.npy'))
            index = np.where((mask == 0) | (mask == 7) | (mask == 8) | (mask == 9) | (mask == 14) | (mask == 15) | (mask == 16) | (mask == 17) | (mask == 18)) # leave face, remove ear
            image_wobg[index[0], index[1], :] = [255, 255, 255]
            image_wobg = Image.fromarray(image_wobg)
            origin_image = image_wobg

        data_dict["origin_image"] = origin_image
        data_dict["image"] = image
        if 'normal' in self.cond:
            data_dict["normal"] = normal
        if 'albedo' in self.cond:
            data_dict["albedo"] = albedo
        if 'rendered' in self.cond:
            data_dict["rendered"] = rendered
        data_dict["prompt"] = prompt
        data_dict["id_feature"] = id_feature

        return data_dict
