import numpy as np
import skimage.transform as skt
import scipy.ndimage.interpolation as sci
import cv2
from scipy.ndimage.filters import gaussian_filter
from warnings import warn

"""
vol: [H, W, D(, C)]

x, y, z <--> W, H, D

you should give (H, W, D) form shape.

skimage interpolation notations:

order = 0: Nearest-Neighbor
order = 1: Bi-Linear (default)
order = 2: Bi-Quadratic 
order = 3: Bi-Cubic
order = 4: Bi-Quartic
order = 5: Bi-Quintic
"""


def rotate(img, angle, axes=(0,1), reshape=False, interpolation=1, border_mode='constant', value=0):
    return sci.rotate(img, angle, axes, reshape=reshape, order=interpolation, mode=border_mode, cval=value)


def shift(img, shift, interpolation=1, border_mode='constant', value=0):
    return sci.shift(img, shift, order=interpolation, mode=border_mode, cval=value)


def crop(img, x1, y1, z1, x2, y2, z2):
    width, height, depth = img.shape[:3]
    if x2 <= x1 or y2 <= y1 or z2 <= z1:
        raise ValueError
    if x1 < 0 or y1 < 0 or z1 < 0:
        raise ValueError
    if x2 > width or y2 > height or z2 > depth:
        img = pad(img, (y2, x2, z2))
        warn('image size smaller than crop size, pad by default.', UserWarning)

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
    width, height, depth = img.shape[:3]
    if height < crop_height or width < crop_width or depth < crop_depth:
        img = pad(img, (crop_width, crop_height, crop_depth))
        warn('image size smaller than crop size, pad by default.', UserWarning)
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

    old_shape = np.array(image.shape[:len(new_shape)])
    new_shape = np.array([max(new_shape[i], old_shape[i]) for i in range(len(new_shape))])

    difference = new_shape - old_shape
    pad_below = difference // 2
    pad_above = difference - pad_below

    pad_list = [list(i) for i in zip(pad_below, pad_above)] + [[0, 0]] * axes_not_pad

    if border_mode == 'reflect':
        res = np.pad(image, pad_list, border_mode)
    elif border_mode == 'constant':
        res = np.pad(image, pad_list, border_mode, constant_values=value)
    else:
        raise ValueError

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
    mn = img.min()
    rng = img.max() - mn
    img = (img - mn)/(rng + eps)
    return np.power(img, gamma)
    
def elastic_transform(image, alpha, sigma, alpha_affine, interpolation=1, border_mode='reflect', approximate=False, random_state=None):
    '''
    apply 2D elastic_transform on each x-y plane
    '''
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape # [H, W, D, C]
    shape_size = shape[:2]
    height, width, depth = shape[:3]

    # Random affine
    center_square = np.float32(shape_size) // 2
    square_size = min(shape_size) // 3
    pts1 = np.float32([center_square + square_size, [center_square[0]+square_size, center_square[1]-square_size], center_square - square_size])
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    M = cv2.getAffineTransform(pts1, pts2)
    
    res = np.zeros_like(image)
    for d in range(depth):
        tmp = image[:, :, d] # [H, W, C]
        tmp = cv2.warpAffine(tmp, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101)
        res[:, :, d] = tmp
    image = res

    # Elastic warp
    if approximate:
        # Approximate computation smooth displacement map with a large enough kernel.
        # On large images (512+) this is approximately 2X times faster
        dx = random_state.rand(height, width).astype(np.float32) * 2 - 1
        cv2.GaussianBlur(dx, (17, 17), sigma, dst=dx)
        dx *= alpha

        dy = random_state.rand(height, width).astype(np.float32) * 2 - 1
        cv2.GaussianBlur(dy, (17, 17), sigma, dst=dy)
        dy *= alpha

    else:
        dx = np.float32(gaussian_filter((random_state.rand(height, width, 1) * 2 - 1), sigma) * alpha)
        dy = np.float32(gaussian_filter((random_state.rand(height, width, 1) * 2 - 1), sigma) * alpha)

    x, y, z = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]), np.arange(shape[2]))
    indices = [np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))]
    
    image = sci.map_coordinates(image, indices, order=interpolation, mode=border_mode).reshape(shape)
    return image

def elastic_transform_2(
    img,
    alpha,
    sigma,
    alpha_affine,
    interpolation=cv2.INTER_LINEAR,
    border_mode=cv2.BORDER_REFLECT_101,
    value=None,
    random_state=42,
    approximate=False,
):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5

    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.
    """
    random_state = np.random.RandomState(random_state)

    height, width, depth = img.shape[:3]

    # Random affine
    center_square = np.float32((height, width)) // 2
    square_size = min((height, width)) // 3
    alpha = float(alpha)
    sigma = float(sigma)
    alpha_affine = float(alpha_affine)

    pts1 = np.float32(
        [
            center_square + square_size,
            [center_square[0] + square_size, center_square[1] - square_size],
            center_square - square_size,
        ]
    )
    pts2 = pts1 + random_state.uniform(-alpha_affine, alpha_affine, size=pts1.shape).astype(np.float32)
    matrix = cv2.getAffineTransform(pts1, pts2)

    # pseoudo 2D
    res = np.zeros_like(img)
    for d in range(depth):
        tmp = img[:, :, d] # [H, W, C]
        tmp = cv2.warpAffine(tmp, M=matrix, dsize=(width, height), flags=interpolation, borderMode=border_mode, borderValue=value)
        res[:, :, d] = tmp
    img = res


    if approximate:
        # Approximate computation smooth displacement map with a large enough kernel.
        # On large images (512+) this is approximately 2X times faster
        dx = random_state.rand(height, width).astype(np.float32) * 2 - 1
        cv2.GaussianBlur(dx, (17, 17), sigma, dst=dx)
        dx *= alpha

        dy = random_state.rand(height, width).astype(np.float32) * 2 - 1
        cv2.GaussianBlur(dy, (17, 17), sigma, dst=dy)
        dy *= alpha
    else:
        dx = np.float32(gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma) * alpha)
        dy = np.float32(gaussian_filter((random_state.rand(height, width) * 2 - 1), sigma) * alpha)

    x, y = np.meshgrid(np.arange(width), np.arange(height))

    map_x = np.float32(x + dx)
    map_y = np.float32(y + dy)

    # pseoudo 2D
    res = np.zeros_like(img)
    for d in range(depth):
        tmp = img[:, :, d] # [H, W, C]
        tmp = cv2.remap(tmp, map1=map_x, map2=map_y, interpolation=interpolation, borderMode=border_mode, borderValue=value)
        res[:, :, d] = tmp
    img = res

    return img

