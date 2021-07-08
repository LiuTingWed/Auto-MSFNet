import os
import cv2
from PIL import Image
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data import Dataset

class MyTestData(Dataset):
    """
    load images for testing
    root: director/to/images/
            structure:
            - root
                - images
                    - images (images here)
                - masks (ground truth)
    """

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    def     __init__(self,img_root,gt_root,test_size,transform=True):
        super(MyTestData, self).__init__()
        self._transform = transform
        self.test_size = test_size
        img_root = img_root
        gt_root = gt_root

        file_names = os.listdir(img_root)
        self.img_names = []
        self.gt_names = []
        self.names = []
        for i, name in enumerate(file_names):
            if not name.endswith('.jpg'):
                continue
            self.img_names.append(
                os.path.join(img_root, name[:-4] + '.jpg')
            )
            self.gt_names.append(
                os.path.join(gt_root,name[:-4] + '.png')
            )
            self.names.append(name[:-4])

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        gt_file = self.gt_names[index]
        gt = Image.open(gt_file).convert('L')
        gt = np.array(gt, dtype=np.int32)
        gt = gt / (gt.max() + 1e-8)
        gt = np.where(gt > 0.5, 1, 0)
        img_file = self.img_names[index]
        img = cv2.imread(img_file)[:,:,::-1].astype(np.float32)
        img = cv2.resize(img, dsize=(self.test_size, self.test_size), interpolation=cv2.INTER_LINEAR)
        name = img_file.split('/')[-1].split('.')[0]

        if self._transform:
            try:
                img,  gt = self.transform(img,gt)
            except ValueError:
                print(name)
            return img, gt,name
        else:
            return img, gt,name

    def transform(self, img,gt):
        img = img.astype(np.float64) / 255
        img -= self.mean
        img /= self.std
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        return img,gt
