import torch
from PIL import Image
from torchvision import transforms
from pycocotools.coco import COCO
from pycocotools import mask as cocmask
import numpy as np
import os
import tqdm

import zipfile
import requests
coco_ann_url = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
class COCOSegDataset(torch.utils.data.Dataset):
    def __init__(self, root:os.path, spilt="val", transform=None, limit_classes=False, num_limit = 100, enable_list=False, cat_list=[], resize = (256,256)
        , preprocess=True):
        valid_split = ["train","val"]
        if spilt in valid_split:
            try:
                self.coco = COCO(os.path.join(root,f"annotations\\instances_{spilt}2017.json"))
            except:
                # annFile doesn't exist, download the annotation file to root.
                print("Annotation file doesn't exists! Downloading")
                if not os.path.exists(root):
                    os.mkdir(root)
                filepath = os.path.join(root,"annotations_trainval2017.zip")
                response = requests.get(coco_ann_url, stream=True)

                # Sizes in bytes.
                total_size = int(response.headers.get("content-length", 0))
                block_size = 1024
                if response.status_code == 200:
                    with tqdm.tqdm(total=total_size, unit="B", unit_scale=True) as progress_bar:
                        with open(filepath, "wb") as file:
                            for data in response.iter_content(block_size):
                                progress_bar.update(len(data))
                                file.write(data)
                else:
                    raise Exception("No annotation file found and failed to download")
                print("Extracting")
                with zipfile.ZipFile(filepath, 'r') as zip_ref:
                    zip_ref.extractall(root)
                print("Annotation file extraction complete")
                self.coco = COCO(os.path.join(root,"annotations\\instances_val2017.json"))
        else:
            raise Exception(f"Invalid spilt: {spilt}")

        self.root = os.path.join(root,spilt+"2017")
        if not os.path.exists(self.root):
            os.mkdir(self.root)
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
            self.__preprocess__(list(self.coco.imgs.keys()), least_pix=1000)
        else:
            self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        image_info = coco.loadImgs(img_id)[0]
        path = image_info['file_name']
        try:
            image = Image.open(os.path.join(self.root, path)).convert('RGB') # Read RGB image
        except:
            ## Image doesn't exists, download from coco_url
            ## This process is quite time consuming and should be avoided later
            response = requests.get(image_info["coco_url"])
            file_Path = os.path.join(self.root,path)
            if response.status_code == 200:
                with open(file_Path, 'wb') as file:
                    file.write(response.content)
                image = Image.open(os.path.join(self.root, path)).convert('RGB')
            else:
                raise Exception(f"No image found for {path} and failed to download")
            
        
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

    def __preprocess__(self,ids,least_pix:int=1000): #Filter out bad samples
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