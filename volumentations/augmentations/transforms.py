import random
import numpy as np
from volumentations.core import Compose, Transform, DualTransform
import volumentations.augmentations.functionals as F

__all__ = [
    "Flip",
    "Normalize",
    "Transpose",
    "RandomCrop",
    "RandomGamma",
    "RandomRotate90",
    "Rotate",
    "ShiftScaleRotate",
    "CenterCrop",
    "ElasticTransform",
    "PadIfNeeded",
    "RandomBrightness",
    "RandomContrast",
    "GaussNoise",
    "ToGray",
    "Crop",
    "CropNonEmptyMaskIfExists",
    "RandomScale",
    "Resize",
]

class PadIfNeeded(DualTransform):
    def __init__(self, shape, border_mode='reflect', value=0, mask_value=0, always_apply=False, p=1):
        super().__init__(always_apply, p)
        self.shape = shape
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img):
        return F.pad(img, self.shape, self.border_mode, self.value)
    
    def apply_to_mask(self, mask):
        return F.pad(mask, self.shape, self.border_mode, self.mask_value)

class GaussNoise(Transform):
    def __init__(self, var_limit=(0, 0.1), mean=0, always_apply=False, p=0.5):
        super(GaussNoise, self).__init__(always_apply, p)
        self.var_limit = var_limit
        self.mean = mean

    def apply(self, img, var):
        return F.gaussian_noise(img, var=var, mean=self.mean)

    def get_params(self, **data):
        return {'var': random.uniform(*self.var_limit)}

class Resize(DualTransform):

    def __init__(self, shape, interpolation=1, always_apply=False, p=1):
        super(Resize, self).__init__(always_apply, p)
        self.shape = shape
        self.interpolation = interpolation

    def apply(self, img):
        return F.resize(img, new_shape=self.shape, interpolation=self.interpolation)
    
    def apply_to_mask(self, mask):
        return F.resize(mask, new_shape=self.shape, interpolation=0)

class RandomScale(DualTransform):

    def __init__(self, scale_limit=[0.9, 1.1], interpolation=1, always_apply=False, p=0.5):
        super(RandomScale, self).__init__(always_apply, p)
        self.scale_limit = scale_limit
        self.interpolation = interpolation

    def get_params(self, **data):
        return {"scale": random.uniform(self.scale_limit[0], self.scale_limit[1])}

    def apply(self, img, scale):
        return F.rescale(img, scale, interpolation=self.interpolation)

    def apply_to_mask(self, mask, scale):
        return F.rescale(mask, scale, interpolation=0)

class Rotate(DualTransform):

    def __init__(self, axes=(0,1), limit=90, interpolation=1, border_mode='reflect', value=0, mask_value=0, always_apply=False, p=0.5):
        super(Rotate, self).__init__(always_apply, p)
        self.axes = axes
        self.limit = limit
        self.interpolation = interpolation
        self.border_mode = border_mode
        self.value = value
        self.mask_value = mask_value

    def apply(self, img, angle):
        return F.rotate(img, angle, axes=self.axes, reshape=False, interpolation=self.interpolation, border_mode=self.border_mode, value=self.value)

    def apply_to_mask(self, mask, angle):
        return F.rotate(mask, angle, axes=self.axes, reshape=False, interpolation=0, border_mode=self.border_mode, value=self.mask_value)

    def get_params(self, **data):
        return {"angle": random.uniform(self.limit[0], self.limit[1])}

class RandomRotate90(DualTransform):

    def __init__(self, axes=(0,1), always_apply=False, p=0.5):
        super(RandomRotate90, self).__init__(always_apply, p)
        self.axes = axes

    def apply(self, img, factor):
        return np.rot90(img, factor, axes=self.axes)

    def get_params(self, **data):
        return {"factor": random.randint(0, 3)}


class Flip(DualTransform):

    def __init__(self, axis=0, always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.axis = axis

    def apply(self, img):
        return np.flip(img, self.axis)

    
class Normalize(Transform):

    def apply(self, img):
        return F.normalize(img)

        
class Transpose(DualTransform):

    def __init__(self, axes=(0,1), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.axes = axes

    def apply(self, img):
        return np.transpose(img, self.axes)

class CenterCrop(DualTransform):

    def __init__(self, shape, always_apply=False, p=1.0):
        super(CenterCrop, self).__init__(always_apply, p)
        self.shape = shape

    def apply(self, img):
        return F.center_crop(img, self.shape[0], self.shape[1], self.shape[2])


class RandomCrop(DualTransform):

    def __init__(self, shape, always_apply=False, p=1.0):
        super(RandomCrop, self).__init__(always_apply, p)
        self.shape = shape

    def apply(self, img, h_start=0, w_start=0, d_start=0):
        return F.random_crop(img, self.shape[0], self.shape[1], self.shape[2], h_start, w_start, d_start)

    def get_params(self, **data):
        return {
            "h_start": random.random(), 
            "w_start": random.random(),
            "d_start": random.random(),
            }
    
class CropNonEmptyMaskIfExists(DualTransform):

    def __init__(self, shape, always_apply=False, p=1.0):
        super(CropNonEmptyMaskIfExists, self).__init__(always_apply, p)
        self.height = shape[0]
        self.width = shape[1]
        self.depth = shape[2]

    def apply(self, img, x_min=0, y_min=0, z_min=0, x_max=0, y_max=0, z_max=0):
        return F.crop(img, x_min, y_min, z_min, x_max, y_max, z_max)

    def get_params(self, **data):
        mask = data["mask"] # [H, W, D]
        mask_height, mask_width, mask_depth = mask.shape

        if mask.sum() == 0:
            x_min = random.randint(0, mask_width - self.width)
            y_min = random.randint(0, mask_height - self.height)
            z_min = random.randint(0, mask_depth - self.depth)
        else:
            non_zero = np.argwhere(mask)
            y, x, z = random.choice(non_zero)
            x_min = x - random.randint(0, self.width - 1)
            y_min = y - random.randint(0, self.height - 1)
            z_min = z - random.randint(0, self.depth - 1)
            x_min = np.clip(x_min, 0, mask_width - self.width)
            y_min = np.clip(y_min, 0, mask_height - self.height)
            z_min = np.clip(z_min, 0, mask_depth - self.depth)

        x_max = x_min + self.width
        y_max = y_min + self.height
        z_max = z_min + self.depth

        return {
            "x_min": x_min, "x_max": x_max,
            "y_min": y_min, "y_max": y_max,
            "z_min": z_min, "z_max": z_max,
            }



class RandomGamma(Transform):

    def __init__(self, gamma_limit=(0.8, 1.2), eps=1e-7, always_apply=False, p=0.5):
        super(RandomGamma, self).__init__(always_apply, p)
        self.gamma_limit = gamma_limit
        self.eps = eps

    def apply(self, img, gamma=1):
        return F.gamma_transform(img, gamma=gamma, eps=self.eps)

    def get_params(self, **data):
        return {"gamma": random.randint(self.gamma_limit[0], self.gamma_limit[1])}