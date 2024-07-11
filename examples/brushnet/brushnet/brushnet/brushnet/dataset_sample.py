from torch.utils.data import ConcatDataset
import os
import os.path
import random
import json
import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import json
from pycocotools.coco import COCO

DATASET_DIR = {
    'COCO': {
        'image_folder': "/lustre/scratch/client/vinai/users/ngannh9/hand_improve/data/COCOWholeBody/train2017",
        'caption_file': "/lustre/scratch/client/vinai/users/ngannh9/hand_improve/data/COCOWholeBody/annotations/coco_wholebody_train_v1.0.json",
        'image_folder_val': "/lustre/scratch/client/vinai/users/ngannh9/hand_improve/data/COCOWholeBody/val2017",
        'caption_file_val': "/lustre/scratch/client/vinai/users/ngannh9/hand_improve/data/COCOWholeBody/annotations/coco_wholebody_val_v1.0.json"
    },
    'handdb': {
        'image_folder': "/lustre/scratch/client/vinai/users/quangnqv/dataset/genai_sb_handfacebody/dataset/DeepFashion_multimodal/images",
        'segmentation_dir': "/lustre/scratch/client/vinai/users/quangnqv/dataset/genai_sb_handfacebody/dataset/DeepFashion_multimodal/segm",
        'caption_file': "/lustre/scratch/client/vinai/users/quangnqv/dataset/genai_sb_handfacebody/dataset/DeepFashion_multimodal/captions.json"
    },
    'laion2m': {
        'image_folder': "/lustre/scratch/client/vinai/users/ngannh9/hand/data/LAION/preprocessed_2256k/train",
        'segmentation_dir': "/lustre/scratch/client/vinai/users/quangnqv/code/Self-Correction-Human-Parsing/outputs/",
        'caption_file': "/lustre/scratch/client/vinai/users/ngannh9/hand/LAVIS/output"
        # 'caption_file': "cap_test"
    },
    'prompt': {
        'prompt_list_path': "/lustre/scratch/client/vinai/users/quangnqv/dataset/genai_sb_handfacebody/dataset/prompt/prompt_list.txt",
        'embed_txt_path': "/lustre/scratch/client/vinai/users/quangnqv/dataset/genai_sb_handfacebody/dataset/prompt/embeds"
    }
}
def center_crop(image, crop_size):
    new_width, new_height = crop_size
    width, height = image.size
    left = (width - new_width) / 2
    top = (height - new_height) / 2
    right = (width + new_width) / 2
    bottom = (height + new_height) / 2
    return image.crop((left, top, right, bottom))


class PromptDataset(data.Dataset):
    def __init__(self, prompt_list_path, embed_txt_path):
        with open(embed_txt_path) as f:
            self.embed_paths = f.readlines()
            self.embed_paths = [x.strip() for x in self.embed_paths]
        with open(prompt_list_path) as f:
            self.prompts = f.readlines()
            self.prompts = [x.strip() for x in self.prompts]

        assert len(self.prompts) == len(
            self.embed_paths
        ), f"Prompt {len(self.prompts)} and embeds {len(self.embed_paths)} length mismatch"

    def _load(self, idx):
        return {
            "prompt_embeds": torch.from_numpy(np.load(self.embed_paths[idx])),
            "prompt": self.prompts[idx],
        }

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, index):
        data = self._load(index)

        return data

    def shuffle(self, *args, **kwargs):
        ids = np.arange(len(self.prompts))
        shuffled_ids = np.random.permutation(ids)
        self.prompts = [self.prompts[idx] for idx in shuffled_ids]
        self.embed_paths = [self.embed_paths[idx] for idx in shuffled_ids]
        return self

    def select(self, selected_range):
        self.prompts = [self.prompts[idx] for idx in selected_range]
        self.embed_paths = [self.embed_paths[idx] for idx in selected_range]
        return self

class DeepFashionTextSegmDataset(data.Dataset):

    def __init__(self,
                 img_dir,
                 segm_dir,
                 caption_path,
                 transform, image_size):
        self._img_path = img_dir
        self._segm_path = segm_dir
        self._caption_path = caption_path
        self._captions = json.load(open(self._caption_path, 'r'))
        self._image_fnames = self.filter_segm_images(list(self._captions.keys()))
        self._caption_path = caption_path
        self.transform = transform
        self.image_size = image_size
        self.resize_mask = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
            transforms.ToTensor(),
        ])
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(self.image_size, interpolation=transforms.InterpolationMode.NEAREST)

    def filter_segm_images(self, image_fnames):
        image_fnames = list(filter(lambda fname: os.path.exists(os.path.join(self._segm_path, f'{fname[:-4]}_segm.png')), image_fnames))
        return image_fnames

    def _open_file(self, path_prefix, fname):
        return open(os.path.join(path_prefix, fname), 'rb')

    def _load_raw_image(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        image = Image.open(os.path.join(self._img_path, fname))
        return image

    def _load_segm(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        fname = f'{fname[:-4]}_segm.png'
        segm = Image.open(os.path.join(self._segm_path, fname))
        return segm

    def _load_captions(self, raw_idx):
        fname = self._image_fnames[raw_idx]
        return self._captions[fname]

    def __getitem__(self, index):
        image = self._load_raw_image(index)
        segm = self._load_segm(index)
        text_desc = self._load_captions(index)
        segm = self.to_tensor(segm) * 255
        segm = torch.logical_or(segm == 15, segm == 14).int() # skin & face

        segm = self.resize(segm).squeeze()
        return_dict = {
            'image': [image],
            'segm': [segm],
            'text': [text_desc],
            'img_name': self._image_fnames[index]
        }

        return self.transform(return_dict)
    

    def __len__(self):
        return len(self._image_fnames)
class CocoHumanDataset(data.Dataset):
    def __init__(self, img_dir, label_dir, image_size, transform):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.infor = self.preprocess()
        self.imgid = list(self.infor.keys())
        self.mask = {}
        self.image_size = image_size
        self.transform = transform
        self.resize_mask = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
        ])
     
    def preprocess(self):
        f = json.load(open(self.label_dir, 'rb'))
        return f
    
    def __len__(self):
        return len(self.infor)
    def __getitem__(self, index):
        """ 
            bbox in here is xywh
        """
        imgid = self.imgid[index]
        infor = self.infor[imgid]
        print(infor)
        img_path = os.path.join(self.img_dir, infor['file_name'])
        caption = infor['caption']
        person_infor = infor['person']
      
        image = Image.open(img_path)
        bbox = [person['lefthand_box'] for person in person_infor] + [person['righthand_box'] for person in person_infor]
        
        height, width = infor['height'], infor['width']
        
        
        mask = np.zeros((height, width))
        for box in bbox:
            x, y, w, h = box
            x, y, w, h = int(x), int(y), int(w), int(h)
            if x==y and y==w and w==h and h==0: continue
            mask[y:y+h, x:x+w] = 1
  
        mask = self.resize_mask(mask).squeeze()
        
        return_dict = {
                'image': [image],
                'text': [caption],
                'img_name': img_path,
                'segm': [mask]
            }
        
        return self.transform(return_dict)
class LaionDataset(torch.utils.data.Dataset):
    def __init__(self, folder_path, text_file_path, mask_path, transform, tokenizer):
        self.folder_path = folder_path
        self.transform = transform
        self.mask_path = mask_path
        self.tokenizer = tokenizer
        self.text_file_path = text_file_path
        self.captions = []
        self.images = []
        self.masks = []
        self.pixel_values = []
        with open(self.text_file_path, 'r') as file:
            for line in file:
                line = line.strip().split(' ')
                image_filename = line[0]
                caption = ' '.join(line[1:])
            
                image_path = os.path.join(self.folder_path, image_filename)
                mask_path = os.path.join(self.mask_path, image_filename)
                if os.path.isfile(mask_path):
                    mask = Image.open(mask_path)
                else:
                    mask = None
                image = Image.open(image_path).convert('RGB')  # Load image as PIL
                pixel_value = self.transform(image)
                self.captions.append(caption)
                self.images.append(image)
                self.masks.append(mask)
                self.pixel_values.append(pixel_value)
        self.input_ids = self.tokenizer(self.captions, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt").input_ids
        self.input_ids = torch.stack([example for example in self.input_ids])
        self.pixel_values = torch.stack([example for example in self.pixel_values])
        self.pixel_values = self.pixel_values.to(memory_format=torch.contiguous_format).float()
    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        return self.pixel_values[idx], self.input_ids[idx]   
class LaiOn2M(data.Dataset):
    def __init__(self, img_dir, label_dir,segmentation_dir, image_size, transform):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.mask = {}
        self.image_size = image_size
        self.transform = transform
        self.seg_dir = segmentation_dir
        self.resize_mask = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
        ])
        # self.infor = self.preprocess()
        # self.imgid = list(self.infor.keys())
        self.filenames, self.captions = self.read_prompt(label_dir)
        # self.infor = self.read_segment(segmentation_dir)
        
        # self.sanity_check()

    def read_prompt(self, label_dir):
        txt_files = os.listdir(label_dir)
        filenames = []
        captions = []
        for txt_file in txt_files:
            with open(os.path.join(label_dir, txt_file), 'r') as file:
                for line in file:
                    line = line.strip().split(' ')
                    filenames.append(line[0])
                    captions.append(' '.join(line[1:]))
        return filenames, captions
    # def read_segment(self, segment_dir):
    #     filenames = []
    #     for file in os.listdir(segment_dir):
    #         if file.endswith('.npy'):
    #             if file.replace('npy', 'jpg') not in self.prompt: continue
    #             filenames.append(file.replace('.npy', '.jpg'))
    #     return filenames

    def sanity_check(self):
        # for file in self.infor:
        #     if file not in self.prompt:
        #         breakpoint()
        pass

        return
    def __len__(self):
        return len(self.filenames)
    def __getitem__(self, index):

        img_name = self.filenames[index]
       
        img_path = os.path.join(self.img_dir, img_name)
        caption = self.captions[index]
      
      
        image = Image.open(img_path)
        seg_path = os.path.join(self.seg_dir, img_name.replace('.jpg', '.npy'))
        if os.path.exists(seg_path):
            mask = np.load(os.path.join(self.seg_dir, img_name.replace('.jpg', '.npy')))
            mask = np.logical_or(mask == 13, mask == 14, mask==15).astype(np.uint8)
        else:
            width, height = image.size
            mask = np.zeros((height, width))
        mask = self.resize_mask(mask).squeeze()
        return_dict = {
                'image': [image],
                'text': [caption],
                'img_name': img_path,
                'segm': [mask]
            }
        
        return self.transform(return_dict)
def get_dataset(dataset_names, transform, image_size):
    dataset_names = dataset_names.split(',')
    dataset_list = []
    for dataset_name in dataset_names:
        if dataset_name == 'COCO':
            _data =  CocoHumanDataset(
                img_dir=DATASET_DIR['COCO']['image_folder'],
                label_dir=DATASET_DIR['COCO']['caption_file'],
                transform=transform,
                image_size=image_size
            )
        elif dataset_name == 'deepfashion_mm':
            _data =  DeepFashionTextSegmDataset(
                img_dir=DATASET_DIR['deepfashion_mm']['image_folder'],
                segm_dir=DATASET_DIR['deepfashion_mm']['segmentation_dir'],
                caption_path=DATASET_DIR['deepfashion_mm']['caption_file'],
                transform=transform,
                image_size=image_size,
            )
        elif dataset_name == 'laion2m':
            _data = LaiOn2M(
                img_dir=DATASET_DIR['laion2m']['image_folder'],
                label_dir=DATASET_DIR['laion2m']['caption_file'],
                segmentation_dir=DATASET_DIR['laion2m']['segmentation_dir'],
                image_size=image_size,
                transform=transform
            )
        elif dataset_name == 'prompt':
            _data = PromptDataset(
                prompt_list_path=DATASET_DIR['prompt']['prompt_list_path'],
                embed_txt_path=DATASET_DIR['prompt']['embed_txt_path']
            )   
        else:
            raise ValueError(f'Unknown dataset name: {dataset_name}')

        dataset_list.append(_data)
    if len(dataset_list) > 1: 
        print("using ", dataset_names)
        return ConcatDataset(dataset_list)
    else:
        return dataset_list[0]

#usage
'''
    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples['image']]
        # print(images[0].size)
        # print(train_transforms(images[0]).shape)
        examples["pixel_values"] = torch.stack([train_transforms(image) for image in images]).squeeze()
        examples["input_ids"] = tokenize_captions(examples['text'])
        examples["segm"] = [mask_transforms(mask) for mask in examples['segm']]
        return examples
        
    train_dataset = get_dataset(
        dataset_names=args.dataset_name,
        transform=preprocess_train,
        image_size=args.resolution
    )
    
    def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = torch.stack([example["input_ids"] for example in examples])
    segms = torch.tensor(np.stack([np.array(example["segm"][0]) for example in examples]))

    return {"pixel_values": pixel_values, "input_ids": input_ids, "mask": segms}

    # DataLoaders creation:
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )
'''
def main():
    resolution = 512
    def preprocess_train(examples):
        images = [image.convert("RGB") for image in examples['image']]
        # print(images[0].size)
        # print(train_transforms(images[0]).shape)
        examples["pixel_values"] = torch.stack([train_transforms(image) for image in images]).squeeze()
        # examples["input_ids"] = tokenize_captions(examples['text'])
        examples["segm"] = [mask_transforms(mask) for mask in examples['segm']]
        return examples
    train_transforms = transforms.Compose(
        [
            # transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(resolution),
            # transforms.ToTensor(),
            # transforms.Normalize([0.5], [0.5]),
        ]
    )
    mask_transforms = transforms.Compose(
        [
            # transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            # transforms.CenterCrop(resolution),
        ]
    )
    train_dataset = get_dataset(
        dataset_names='COCO',
        transform=preprocess_train,
        image_size=resolution
    )
    print(train_dataset[0])
    
if __name__ == "__main__":
    main()
