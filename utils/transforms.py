import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

class Transforms:
    def __init__(self, means, stds):
        # Convert numpy arrays to lists if necessary
        if isinstance(means, np.ndarray):
            means = means.tolist()
        if isinstance(stds, np.ndarray):
            stds = stds.tolist()
            
        self.train_transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.5),
            A.CoarseDropout(
                max_holes=1, max_height=16, max_width=16,
                min_holes=1, min_height=16, min_width=16,
                fill_value=[x * 255 for x in means],  # Convert to 0-255 range
                p=0.5
            ),
            A.Normalize(mean=means, std=stds),
            ToTensorV2()
        ])

        self.test_transform = A.Compose([
            A.Normalize(mean=means, std=stds),
            ToTensorV2()
        ])

    def __call__(self, img, train=True):
        # Convert PIL Image to numpy array if needed
        if not isinstance(img, np.ndarray):
            img = np.array(img)
            
        if train:
            return self.train_transform(image=img)["image"]
        return self.test_transform(image=img)["image"] 