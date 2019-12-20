import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from volumentations import *
import SimpleITK as sitk
import nibabel as nib

def view_batch(imgs, lbls):
    '''
    imgs: [D, H, W, C], the depth or batch dimension should be the first.
    '''
    fig = plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    ax1.set_title('image')
    ax2.set_title('label')
    """
    if init with zeros, the animation may not update? seems bug in animation.
    """
    img1 = ax1.imshow(np.random.rand(*imgs.shape[1:]))
    img2 = ax2.imshow(np.random.rand(*lbls.shape[1:]))
    def update(i):
        plt.suptitle(str(i))
        img1.set_data(imgs[i])
        img2.set_data(lbls[i])
        return img1, img2
    ani = animation.FuncAnimation(fig, update, frames=len(imgs), interval=10, blit=False, repeat_delay=0)
    plt.show()

patch_size = (32, 160, 256)

#img = nib.load('data/img.nii').get_fdata()
#lbl = nib.load('data/lbl.nii').get_fdata()

img = sitk.ReadImage('data/img.nii')
img = sitk.GetArrayFromImage(img)
lbl = sitk.ReadImage('data/lbl.nii')
lbl = sitk.GetArrayFromImage(lbl)

# add grids
"""
h, w, d = img.shape
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
        RemoveEmptyBorder(always_apply=True),
        RandomScale((0.8, 1.2)),
        PadIfNeeded(patch_size, always_apply=True),
        #RandomCrop(patch_size, always_apply=True),
        #CenterCrop(patch_size, always_apply=True),
        #RandomCrop(patch_size, always_apply=True),
        #Resize(patch_size, always_apply=True),
        CropNonEmptyMaskIfExists(patch_size, always_apply=True),
        Normalize(always_apply=True),
        #ElasticTransform((0, 0.25)),
        #Rotate((-15,15),(-15,15),(-15,15)),
        #Flip(0),
        #Flip(1),
        #Flip(2),
        #Transpose((1,0,2)), # only if patch.height = patch.width
        #RandomRotate90((0,1)),
        #RandomGamma(),
        #GaussianNoise(),
    ], p=1)

aug = get_augmentation()

data = {
        'image': img,
        'mask': lbl,
        }

for i in range(10):
    aug_data = aug(**data)
    img, lbl = aug_data['image'], aug_data['mask']
    print(img.shape, lbl.shape, np.max(img), np.max(lbl))
    view_batch(img, lbl)
