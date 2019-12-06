# Volumentations

3D Volume data augmentation package inspired by `albumentations`

### Simple Example

```python
def get_augmentation(patch_size):
    return Compose([
        Resize(patch_size, always_apply=True),
        #CropNonEmptyMaskIfExists(patch_size, always_apply=True),
        Normalize(always_apply=True),
        ElasticTransform((0, 0.25)),
        Rotate((-15,15),(-15,15),(-15,15)),
        #Flip(0),
        #Flip(1),
        #Flip(2),
        #Transpose((1,0,2)), # need patch.height = patch.width
        #RandomRotate90((0,1)),
        RandomGamma(),
        GaussianNoise(),
    ], p=0.8)

aug = get_augmentation()

data = {'image': img, 'mask': lbl}
aug_data = aug(**data)
img, lbl = aug_data['image'], aug_data['mask']
```

### Implemented Augmentations

```python
class PadIfNeeded(DualTransform):
class GaussianNoise(Transform):
class Resize(DualTransform):
class RandomScale(DualTransform):
class RotatePseudo2D(DualTransform):
class RandomRotate90(DualTransform):
class Flip(DualTransform):
class Normalize(Transform):
class Float(DualTransform):
class Contiguous(DualTransform):
class Transpose(DualTransform):
class CenterCrop(DualTransform):
class RandomResizedCrop(DualTransform):
class RandomCrop(DualTransform):
class CropNonEmptyMaskIfExists(DualTransform):
class ResizedCropNonEmptyMaskIfExists(DualTransform):
class RandomGamma(Transform):
class ElasticTransformPseudo2D(DualTransform):
class ElasticTransform(DualTransform):
class Rotate(DualTransform):
```

