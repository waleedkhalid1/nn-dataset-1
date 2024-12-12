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

def loader(path="./data/cocos",resize=(128,128), **kwargs):
    train_set = COCOSegDataset(root=path,spilt="train",resize=resize,preprocess=True,**kwargs)
    val_set = COCOSegDataset(root=path,spilt="val",resize=resize,preprocess=True,**kwargs)
    return train_set, val_set


class COCOSegDataset(torch.utils.data.Dataset):
    def __init__(self, root:os.path, spilt="val", transform=transforms.Compose([transforms.ToTensor()]), class_limit = None, num_limit = None, class_list=None, resize = (128,128)
        , preprocess=True, least_pix=1000, **kwargs):
        """Read datas from COCOS and generate 2D masks.

        Parameters
        ----------
        path : Path towards cocos root directory. It should be structured as default.
        spilt : str `"train"` or `"val"`.
        transform : transform towards the image. For resizing, please use `resize` parameter,
          for torch transforms might have issues transforming masks.
        class_limit : Limit class index from 0 to the value. Set to `None` for no limit.
        num_limit : Limit maximum number of images to use. Only works with `preprocess`.
        class_list : Limit class index within the list. Set to `None` for no limit,
          It will ignore the `class_limit` parameter.
        resize : tuple (h,w) to resize the image and its mask. Uses Image from PIL to avoid
          artifacts on the mask
        preprocess : Set true to allow preprocess that filter out all images with mask that
          have lesser than `least_pix` pixels.
        least_pix : filter out thersold of preprocess.
        """
        valid_split = ["train","val"]
        if spilt in valid_split:
            try:
                self.coco = COCO(os.path.join(root,"annotations",f"instances_{spilt}2017.json"))
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
                self.coco = COCO(os.path.join(root,"annotations",f"instances_{spilt}2017.json"))
        else:
            raise Exception(f"Invalid spilt: {spilt}")

        self.root = os.path.join(root,spilt+"2017")
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        self.transform = transform
        self.num_classes = len(self.coco.getCatIds())
        self.masks = []
        self.resize = resize
        self.limit_classes = class_limit
        self.num_limit = num_limit
        self.class_list = class_list
        self.no_missing_img = False
        self.mask_transform = transforms.Compose([transforms.ToTensor()])
        if preprocess:
            self.ids = []
            self.__preprocess__(list(self.coco.imgs.keys()), least_pix=least_pix)
        else:
            self.ids = list(self.coco.imgs.keys())

    def __getitem__(self, index):
        coco = self.coco
        img_id = self.ids[index]
        image_info = coco.loadImgs(img_id)[0]
        file_Path = os.path.join(self.root,image_info['file_name'])
        try:
            image = Image.open(file_Path).convert('RGB') # Read RGB image
        except:
            ## Image doesn't exists, download from coco_url
            ## This process is quite time consuming and should be avoided later
            if self.no_missing_img:
                print("Failed to read image(s). Download will be performed thereafter, but it can significantly slowdown processing.")
                self.no_missing_img= True
            response = requests.get(image_info["coco_url"])
            if response.status_code == 200:
                with open(file_Path, 'wb') as file:
                    file.write(response.content)
                image = Image.open(file_Path).convert('RGB')
            else:
                raise Exception(f"No image found for {image_info['file_name']} and failed to download")
            
        
        # Create mask
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        mask = np.zeros((image_info["height"], image_info["width"]))
        for ann in anns:
            rle = cocmask.frPyObjects(ann['segmentation'], image_info["height"], image_info["width"])
            m = cocmask.decode(rle)
            cat = ann['category_id']
            if self.class_list==None and self.limit_classes==None:
                c = cat
            elif not self.class_list==None and cat in self.class_list:
                c = self.class_list.index(cat)
            elif not self.limit_classes==None and ann['category_id'] <= self.limit_classes:
                c = cat
            else:
                continue
            if len(m.shape) < 3:
                mask[:, :] += (mask == 0) * (m * c)
            else:
                mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
        image = image.resize(size=self.resize, resample=Image.BILINEAR)
        mask = Image.fromarray(mask).resize(size=self.resize, resample=Image.NEAREST)
        image = self.transform(image)
        mask = self.mask_transform(mask).long()
        return image, mask[0]

    def __preprocess__(self,ids,least_pix:int=1000): #Filter out bad samples
        coco = self.coco
        tbar = tqdm.trange(len(ids))
        for i in tbar:
            if not self.num_limit==None and len(self.ids)+1>self.num_limit:
                print("num_limit exceeded, abort preprocess")
                break
            img_id = ids[i]
            image_info = coco.loadImgs(img_id)[0]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            mask = np.zeros((image_info["height"], image_info["width"]))
            for ann in anns:
                rle = cocmask.frPyObjects(ann['segmentation'], image_info["height"], image_info["width"])
                m = cocmask.decode(rle)
                cat = ann['category_id']
                if self.class_list==None and self.limit_classes==None:
                    c = cat
                elif not self.class_list==None and cat in self.class_list:
                    c = self.class_list.index(cat)
                elif not self.limit_classes==None and ann['category_id'] <= self.limit_classes:
                    c = cat
                else:
                    continue
                if len(m.shape) < 3:
                    mask[:, :] += (mask == 0) * (m * c)
                else:
                    mask[:, :] += (mask == 0) * (((np.sum(m, axis=2)) > 0) * c).astype(np.uint8)
            if (mask > 0).sum() > least_pix:
                self.ids.append(img_id)
                file_Path = os.path.join(self.root,image_info['file_name'])
                if not os.path.exists(file_Path):
                    if self.no_missing_img:
                        print("Noticed missing image(s). Download will be performed thereafter, but it can significantly slowdown processing.")
                        self.no_missing_img= True
                    response = requests.get(image_info["coco_url"])
                    if response.status_code == 200:
                        with open(file_Path, 'wb') as file:
                            file.write(response.content)
                    else:
                        raise Exception(f"Failed to download image:{image_info['file_name']}")
            tbar.set_description(f"Accepted {len(self.ids)} of {i+1} images.")

    def __len__(self):
        return len(self.ids)