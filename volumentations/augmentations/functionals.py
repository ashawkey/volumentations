import numpy as np
import skimage.transform as skt
import scipy.ndimage.interpolation as sci

"""
assert vol: [H, W, D(, C)]

skimage interpolation notations:

order = 0: Nearest-Neighbor
order = 1: Bi-Linear (default)
order = 2: Bi-Quadratic 
order = 3: Bi-Cubic
order = 4: Bi-Quartic
order = 5: Bi-Quintic
"""


def rotate(img, angle, axes=(0,1), reshape=True, interpolation=1, border_mode='constant', value=0):
    return sci.rotate(img, angle, axes, reshape=reshape, order=interpolation, mode=border_mode, cval=value)


def shift(img, shift, interpolation=1, border_mode='constant', value=0):
    return sci.shift(img, shift, order=interpolation, mode=border_mode, cval=value)


def crop(img, x1, y1, z1, x2, y2, z2):
    height, width, depth = img.shape[:3]
    if x2 <= x1 or y2 <= y1 or z2 <= z1:
        raise ValueError
    if x1 < 0 or x2 > width or y1 < 0 or y2 > height or z1 < 0 or z2 > depth:
        raise ValueError

    return img[y1:y2, x1:x2, z1:z2]


def get_center_crop_coords(height, width, depth, crop_height, crop_width, crop_depth):
    y1 = (height - crop_height) // 2
    y2 = y1 + crop_height
    x1 = (width - crop_width) // 2
    x2 = x1 + crop_width
    z1 = (depth - crop_depth) // 2
    z2 = z1 + crop_depth
    return x1, y1, z1, x2, y2, z2


def center_crop(img, crop_height, crop_width, crop_depth):
    height, width, depth = img.shape[:3]
    if height < crop_height or width < crop_width or depth < crop_depth:
        raise ValueError
    x1, y1, z1, x2, y2, z2 = get_center_crop_coords(height, width, depth, crop_height, crop_width, crop_depth)
    img = img[y1:y2, x1:x2, z1:z2]
    return img


def get_random_crop_coords(height, width, depth, crop_height, crop_width, crop_depth, h_start, w_start, d_start):
    y1 = int((height - crop_height) * h_start)
    y2 = y1 + crop_height
    x1 = int((width - crop_width) * w_start)
    x2 = x1 + crop_width
    z1 = int((depth - crop_depth) * d_start)
    z2 = z1 + crop_depth
    return x1, y1, z1, x2, y2, z2


def random_crop(img, crop_height, crop_width, crop_depth, h_start, w_start, d_start):
    height, width, depth = img.shape[:3]
    if height < crop_height or width < crop_width or depth < crop_depth:
        raise ValueError
    x1, y1, z1, x2, y2, z2 = get_random_crop_coords(height, width, depth, crop_height, crop_width, crop_depth, h_start, w_start, d_start)
    img = img[y1:y2, x1:x2, z1:z2]
    return img


def normalize(img):
    img = img.astype(np.float32)
    mean = img.mean()
    std = img.std()
    denominator = np.reciprocal(std)
    img = (img - mean) * denominator
    return img

def pad(image, new_shape, border_mode="constant", value=0):
    '''
    image: [H, W, D, C] or [H, W, D]
    new_shape: [H, W, D]
    '''
    axes_not_pad = len(image.shape) - len(new_shape)

    old_shape = np.array(image.shape[len(new_shape):])
    new_shape = np.array([max(new_shape[i], old_shape[i]) for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference - pad_below

    pad_list = [list(i) for i in zip(pad_below, pad_above)] + [[0, 0]] * axes_not_pad

    res = np.pad(image, pad_list, border_mode, constant_values=value)

    return res

def gaussian_noise(img, var, mean):
    return img + np.random.normal(mean, var, img.shape)

def resize(img, new_shape, interpolation=3):
    """
    img: [H, W, D, C] or [H, W, D]
    new_shape: [H, W, D]
    """
    return skt.resize(img, new_shape, order=interpolation, mode='constant', cval=0, clip=True, anti_aliasing=False)

def rescale(img, scale, interpolation=1):
    """
    img: [H, W, D, C] or [H, W, D]
    scale: scalar float
    """
    return skt.rescale(img, scale, order=interpolation, mode='constant', cval=0, clip=True, multichannel=True, anti_aliasing=False)

def gamma_transform(img, gamma, eps=1e-7):
    return np.power(img, gamma)
    