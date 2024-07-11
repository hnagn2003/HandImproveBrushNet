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
import cv2
import datasets
import jsonlines

DATASET_DIR = {
    'COCO': {
        'image_folder': "/lustre/scratch/client/vinai/users/ngannh9/hand/data/COCOWholeBody/train2017",
        'caption_file': "/lustre/scratch/client/vinai/users/ngannh9/coco_wholebody_train_v1.0.json",
        'mask_folder': "/lustre/scratch/client/vinai/users/ngannh9/hand/gen_mask/output/coco_mask"
        # 'image_folder_val': "/lustre/scratch/client/vinai/users/ngannh9/hand_improve/data/COCOWholeBody/val2017",
        # 'caption_file_val': "/lustre/scratch/client/vinai/users/ngannh9/hand_improve/data/COCOWholeBody/annotations/coco_wholebody_val_v1.0.json"
    },
    'hico': {
        'image_folder': "/lustre/scratch/client/vinai/users/ngannh9/hand/data/halpe/hico_20160224_det/images/train2015",
        # 'segmentation_dir': "/lustre/scratch/client/vinai/users/quangnqv/dataset/genai_sb_handfacebody/dataset/DeepFashion_multimodal/segm",
        'caption_file': "/lustre/scratch/client/vinai/users/ngannh9/hand/data/halpe/hico_20160224_det/halpe_hico_caption.jsonl",
        'mask_folder': "/lustre/scratch/client/vinai/users/ngannh9/hand/gen_mask/output/halpe_mask"
    },
    '13k': {
        'image_folder': "/lustre/scratch/client/vinai/users/ngannh9/hand/data/hand13k/training_dataset/training_data/images",
        'caption_file': "/lustre/scratch/client/vinai/users/ngannh9/hand/gen_text/llava/output/13khand/captions_4069_sorted.jsonl",
        'mask_folder': "/lustre/scratch/client/vinai/users/ngannh9/hand/gen_mask/output/halpe_mask"
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

# class CocoHumanDataset(data.Dataset):
#     def __init__(self, img_dir, label_dir, image_size, transform):
#         self.img_dir = img_dir
#         self.label_dir = label_dir
#         self.infor = self.preprocess()
#         self.images = (self.infor['images'])
#         self.annotations = self.infor['annotations']
#         self.info = self.infor['info']
#         print((self.info))
#         self.mask = {}
#         self.image_size = image_size
#         self.transform = transform
#         self.resize_mask = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
#         ])
     
#     def preprocess(self):
#         f = json.load(open(self.label_dir, 'rb'))
#         return f
    
#     def __len__(self):
#         return len(self.infor)
#     def __getitem__(self, index):
#         """ 
#             bbox in here is xywh
#         """
#         image = self.images[index]
#         imgid = image['id']
#         # infor = self.infor[imgid]
#         img_path = os.path.join(self.img_dir, image['file_name'])
#         caption = self.info[index]['description']      
#         image = Image.open(img_path)
#         # bbox = [person['lefthand_box'] for person in person_infor] + [person['righthand_box'] for person in person_infor]
#         height, width = image['height'], image['width']
#         print(self.infor[index])
#         exit()
        
#         mask = np.zeros((height, width))
#         # for box in bbox:
#         #     x, y, w, h = box
#         #     x, y, w, h = int(x), int(y), int(w), int(h)
#         #     if x==y and y==w and w==h and h==0: continue
#         #     mask[y:y+h, x:x+w] = 1
  
#         mask = self.resize_mask(mask).squeeze()
        
#         return_dict = {
#                 'image': [image],
#                 'text': [caption],
#                 'img_name': img_path,
#                 'segm': [mask]
#             }
        
#         return self.transform(return_dict)

# class CocoHumanDataset(data.Dataset):
#     def __init__(self, img_dir, label_dir, image_size, transform):
#         self.img_dir = img_dir
#         self.label_dir = label_dir
#         self.infor = self.preprocess()
#         self.imgid = list(self.infor.keys())
#         self.mask = {}
#         self.image_size = image_size
#         self.transform = transform
#         self.resize_mask = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
#         ])
     
#     def preprocess(self):
#         f = json.load(open(self.label_dir, 'rb'))
#         return f
    
#     def __len__(self):
#         return len(self.infor)
#     def __getitem__(self, index):
#         """ 
#             bbox in here is xywh
#         """
#         imgid = self.imgid[index]
#         infor = self.infor[imgid]
#         img_path = os.path.join(self.img_dir, infor['file_name'])
#         caption = infor['caption']
#         person_infor = infor['person']
      
#         image = cv2.imread(img_path).astype(np.uint8)
#         bbox = [person['lefthand_box'] for person in person_infor] + [person['righthand_box'] for person in person_infor]
#         kpts = []
#         for person in person_infor:
#             if person['lefthand_valid']:
#                 kpts.append(person['lefthand_kpts'])
#             if person['righthand_valid']:
#                 kpts.append(person['righthand_kpts'])
#         height, width = infor['height'], infor['width']
#         mask = np.zeros((height, width, 1)).astype(np.uint8)
#         for box in bbox:
#             x, y, w, h = box
#             x, y, w, h = int(x), int(y), int(w), int(h)
#             if x==y and y==w and w==h and h==0: continue
#             mask[y:y+h, x:x+w] = [1]
#         return_dict = {
#                 'image': image,
#                 'height' : height,
#                 'width' : width,
#                 'caption': caption,
#                 'img_name': img_path,
#                 'mask': mask,
#                 'kpts' : kpts
#             }
        
#         return self.transform(return_dict)
# class CocoHumanDataset(data.Dataset):
#     def __init__(self, img_dir, label_dir, image_size, transform):
#         self.img_dir = img_dir
#         self.label_dir = label_dir
#         self.infor = self.preprocess()
#         self.imgid = list(self.infor.keys())
#         self.mask = {}
#         self.image_size = image_size
#         self.transform = transform
#         self.resize_mask = transforms.Compose([
#             transforms.ToTensor(),
#             transforms.Resize(image_size, interpolation=transforms.InterpolationMode.NEAREST),
#         ])
     
#     def preprocess(self):
#         f = json.load(open(self.label_dir, 'rb'))
#         return f
    
#     def __len__(self):
#         return len(self.infor)
#     def __getitem__(self, index):
#         """ 
#             bbox in here is xywh
#         """
#         imgid = self.imgid[index]
#         infor = self.infor[imgid]
#         img_path = os.path.join(self.img_dir, infor['file_name'])
#         caption = infor['caption']
#         person_infor = infor['person']
      
#         image = cv2.imread(img_path).astype(np.uint8)
#         bbox = [person['lefthand_box'] for person in person_infor] + [person['righthand_box'] for person in person_infor]
#         kpts = []
#         for person in person_infor:
#             if person['lefthand_valid']:
#                 kpts.append(person['lefthand_kpts'])
#             if person['righthand_valid']:
#                 kpts.append(person['righthand_kpts'])
#         height, width = infor['height'], infor['width']
#         mask = np.zeros((height, width, 1)).astype(np.uint8)
#         for box in bbox:
#             x, y, w, h = box
#             x, y, w, h = int(x), int(y), int(w), int(h)
#             if x==y and y==w and w==h and h==0: continue
#             mask[y:y+h, x:x+w] = [1]
#         return_dict = {
#                 'image': image,
#                 'height' : height,
#                 'width' : width,
#                 'caption': caption,
#                 'img_name': img_path,
#                 'mask': mask,
#                 'kpts' : kpts
#             }
        
#         return self.transform(return_dict)

#Note that Hico and 13k has the same format
class HicoDataset(datasets.Dataset):
    def __init__(self, image_folder, caption_file, mask_path, image_size, transform):
        self.image_folder = image_folder
        self.caption_file = caption_file
        self.mask_path = mask_path
        self.transform = transform
        self.image_size = image_size
        self.image_files = []
        self.image_captions = []
        with jsonlines.open(mask_path) as reader:
            for line in reader:
                img_path = line['img_path']
                caption = line['caption'].lower()
                self.image_files.append(img_path)
                self.image_captions.append(caption)

    def __len__(self):
        return len(self.image_files)
    def __getitem__(self, index):
        image_names = self.image_files[index]
        image_paths = [os.path.join(self.image_folder, image_name) for image_name in image_names]
        mask_paths = [os.path.join(self.mask_path, image_name) for image_name in image_names]
        captions = self.image_captions[index]
        images = [Image.open(image_path) for image_path in image_paths]
        masks = [torch.from_numpy(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)).float() for mask_path in mask_paths]
        res = {
            'image' : images,
            'img_name' : image_names,
            'mask': masks,
            'text': captions
        }
        return res
    
class CocoDataset(data.Dataset):
    def __init__(self, img_dir, label_dir, mask_path, image_size, transform):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.mask_path = mask_path

        self.infor = self.preprocess()
        self.imgid = list(self.infor.keys())
        self.image_size = image_size
        self.transform = transform
     
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
        img_path = os.path.join(self.img_dir, infor['file_name'])
        caption = infor['caption']
      
        image = Image.open(img_path)
        mask_paths = [os.path.join(self.mask_path, image_name) for image_name in img_path]

        masks = [torch.from_numpy(cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)).float() for mask_path in mask_paths]

        return_dict = {
            'image': [image],
            'text': [caption],
            'img_name': img_path,
            'mask': masks
        }
        return self.transform(return_dict)

def tokenize_captions(self, caption, tokenizer, is_train=True):
    if random.random() < 0:
        caption=""
    elif isinstance(caption, str):
        caption=caption
    elif isinstance(caption, (list, np.ndarray)):
        # take a random caption if there are multiple
        caption=random.choice(caption) if is_train else caption[0]
    else:
        raise ValueError(
            f"Caption column `{caption}` should contain either strings or lists of strings."
        )
    inputs = tokenizer(
        caption, max_length=tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    return inputs.input_ids

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
        print(images[0].size)
        print(train_transforms(images[0]).shape)
        examples["pixel_values"] = torch.stack([train_transforms(image) for image in images]).squeeze()
        examples["input_ids"] = tokenize_captions(examples['text'])
        examples["segm"] = [mask_transforms(mask) for mask in examples['segm']]
        return examples
    train_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    mask_transforms = transforms.Compose(
        [
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(resolution),
        ]
    )
    train_dataset = get_dataset(
        dataset_names='COCO',
        transform=preprocess_train,
        image_size=resolution
    )
    def collate_fn(examples):
        pixel_values=[]
        conditioning_pixel_values=[]
        masks=[]
        input_ids=[]
        for example in examples:
            caption = example['caption']
            height = example['height']
            width = example['width']
            image = example['image']
            mask = example['mask']
            masked_image=image*mask

            if width>height:
                scale=resolution/height          
            else:
                scale=resolution/width
            w_new=int(np.ceil(width*scale))
            h_new=int(np.ceil(height*scale))
            image=cv2.resize(image,(h_new,w_new),interpolation=cv2.INTER_CUBIC)
            masked_image=cv2.resize(masked_image,(h_new,w_new),interpolation=cv2.INTER_CUBIC)
            mask=cv2.resize(mask,(h_new,w_new),interpolation=cv2.INTER_CUBIC)[:,:,np.newaxis]

            random_crop=[random.randint(0,w_new-resolution),random.randint(0,h_new-resolution)]

            image=image[random_crop[0]:random_crop[0]+resolution,random_crop[1]:random_crop[1]+resolution,:]
            masked_image=masked_image[random_crop[0]:random_crop[0]+resolution,random_crop[1]:random_crop[1]+resolution,:]
            mask=mask[random_crop[0]:random_crop[0]+resolution,random_crop[1]:random_crop[1]+resolution,:]
            
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            masked_image = cv2.cvtColor(masked_image, cv2.COLOR_BGR2RGB)
            image = (image.astype(np.float32) / 127.5) - 1.0
            masked_image = (masked_image.astype(np.float32) / 127.5) - 1.0

            mask=mask.astype(np.float32)

            pixel_values.append(torch.tensor(image).permute(2,0,1))
            conditioning_pixel_values.append(torch.tensor(masked_image).permute(2,0,1))
            masks.append(torch.tensor(mask).permute(2,0,1))
            # input_ids.append(tokenize_captions(caption, tokenizer)[0])
            input_ids.append(0)

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        conditioning_pixel_values = torch.stack(conditioning_pixel_values)
        conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()
        masks = torch.stack(masks)
        masks = masks.to(memory_format=torch.contiguous_format).float()
        input_ids = torch.stack(input_ids)

        return {
            "pixel_values": pixel_values,
            "conditioning_pixel_values": conditioning_pixel_values,
            "masks":masks,
            "input_ids": input_ids,
            }
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=2,
        num_workers=8
    )
    for batch in train_dataloader:
        print(batch)    
if __name__ == "__main__":
    main()
