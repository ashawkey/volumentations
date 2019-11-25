import numpy as np
from batchviewer import view_batch

from volumentations.core import Compose
from volumentations.augmentations.transforms import *

img = np.load('data/img.npy')
lbl = np.load('data/lbl.npy')

h, w, d = img.shape

"""
# grid
for i in range(0, h, 100):
    img[i:i+10, :, :] = 10
    lbl[i:i+10, :, :] = 1
for i in range(0, w, 100):
    img[:, i:i+10, :] = 10
    lbl[:, i:i+10, :] = 1
for i in range(30, d-1, 30):
    img[:, :, i:i+5] = 10
    lbl[:, :, i:i+5] = 1
"""

print(img.shape, lbl.shape)
#view_batch(img.transpose(2,0,1), lbl.transpose(2,0,1))

def get_augmentation():
    return Compose([
        #RandomScale((0.8, 1.2)),
        PadIfNeeded((160,160,64), always_apply=True),
        #RandomCrop((160,160,64), always_apply=True),
        #RandomResizedCrop((160,160,64), (0.8, 2), always_apply=True),
        #Resize((160,160,64), always_apply=True),
        #CropNonEmptyMaskIfExists((160,160,64), always_apply=True),
        ResizedCropNonEmptyMaskIfExists((160,160,64), (0.8, 1.2), always_apply=True),
        Normalize(always_apply=True),
        Rotate((0,1), (-90,90)),
        Rotate((1,2), (-5,5), p=0.2), # reflect will cause stange border
        Rotate((0,2), (-5,5), p=0.2),
        Flip(0),
        Flip(1),
        Flip(2),
        Transpose((1,0,2)), # only if patch.height = patch.width
        RandomRotate90((0,1)),
        RandomGamma(),
        #GaussianNoise(),
        #RandomCrop(),
        #CenterCrop(),
        #ElasticTransform(always_apply=True), # do not use
    ], p=1)

aug = get_augmentation()

data = {
        'volume': img,
        'mask': lbl,
        }

for i in range(10):
    aug_data = aug(**data)
    img, lbl = aug_data['volume'], aug_data['mask']
    print(img.shape, lbl.shape, np.max(img), np.max(lbl))
    view_batch(img.transpose(2,0,1), lbl.transpose(2,0,1))
