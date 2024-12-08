import torch
from PIL import Image
from torchvision import transforms
from pycocotools.coco import COCO
from pycocotools import mask as cocmask
import numpy as np
import os
import tqdm

class COCOSegDataset(torch.utils.data.Dataset):
    def __init__(self, root, annFile, transform=transforms.Compose([transforms.ToTensor()]), limit_classes:bool=False, num_limit:int = 100, enable_list:bool=False, cat_list=[], resize = (128,128)
        , preprocess:bool=True, leastPix:int = 1000):
        self.coco = COCO(annFile)
        self.root = root
        self.transform = transform
        self.num_classes = len(self.coco.getCatIds())
        self.masks = []
        self.resize = resize
        self.limit_classes = limit_classes
        self.num_limit = num_limit
        self.enable_list = enable_list
        self.cat_list = cat_list
        if preprocess:
            self.ids = []
            self.__preprocess__(list(self.coco.imgs.keys()), least_pix=leastPix)
        else:
            self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        image_info = coco.loadImgs(img_id)[0]
        path = image_info['file_name']
        image = Image.open(os.path.join(self.root, path)).convert('RGB') # Read RGB image
        
        # Create mask
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        mask = np.zeros((image_info["height"], image_info["width"]))
        for ann in anns:
            rle = cocmask.frPyObjects(ann['segmentation'], image_info["height"], image_info["width"])
            m = cocmask.decode(rle)
            cat = ann['category_id']
            if not self.enable_list and not self.limit_classes:
                c = cat
            elif self.enable_list and cat in self.cat_list:
                c = self.cat_list.index(cat)
            elif self.limit_classes and ann['category_id'] <= self.num_limit:
                c = cat
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        image = image.resize(size=self.resize, resample=Image.BILINEAR)
        mask = Image.fromarray(mask).resize(size=self.resize, resample=Image.NEAREST)
        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask).long()
        return image, mask[0]

    def __preprocess__(self,ids,least_pix:int=1000): #Filter out empty samples
        coco = self.coco
        tbar = tqdm.trange(len(ids))
        for i in tbar:
            img_id = ids[i]
            image_info = coco.loadImgs(img_id)[0]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            mask = np.zeros((image_info["height"], image_info["width"]))
            for ann in anns:
                rle = cocmask.frPyObjects(ann['segmentation'], image_info["height"], image_info["width"])
                m = cocmask.decode(rle)
                cat = ann['category_id']
                if not self.enable_list and not self.limit_classes:
                    c = cat
                elif self.enable_list and cat in self.cat_list:
                    c = self.cat_list.index(cat)
                elif self.limit_classes and ann['category_id'] <= self.num_limit:
                    c = cat
                else:
                    continue
                if len(m.shape) < 3:
                    mask[:, :] += (mask == 0) * (m * c)
                else:
                    mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
            if (mask > 0).sum() > least_pix:
                self.ids.append(img_id)
            tbar.set_description(f"Accepted {len(self.ids)} of {i+1} images.")

    def __len__(self):
        return len(self.ids)