import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class StyleDataset(Dataset):
    """Load dataset with images and captions stored together"""
    def __init__(self, data_path: str, center_crop: bool = False, image_size: int = 512):
        self.data_path = data_path

        # Only consider images (captions will be matched by basename)
        self.image_names = [f for f in os.listdir(data_path) if f.lower().endswith('.jpg')]


        # self.image_transforms = transforms.Compose([
        #     transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        #     # transforms.CenterCrop(image_size) if center_crop else transforms.RandomCrop(256),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        # ])


        trans = [
            # random resized crop gives scale & aspect augment
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.25, hue=0.02)], p=0.8),
            transforms.RandomApply([transforms.RandomRotation(degrees=10)], p=0.5),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=(3, 5))], p=0.3),
            # small affine for subtle geometric perturbations
            transforms.RandomAffine(degrees=6, translate=(0.03, 0.03), scale=(0.98, 1.02), shear=2),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]

        self.image_transforms = transforms.Compose(trans)

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx):
        # Get image
        image_name = self.image_names[idx]
        image_path = os.path.join(self.data_path, image_name)

        im = Image.open(image_path)  # ensure RGB
        im = self.image_transforms(im)

        # Get corresponding caption
        caption_name = os.path.splitext(image_name)[0] + ".txt"
        caption_path = os.path.join(self.data_path, caption_name)

        with open(caption_path, "r") as f:
            caption = f.read().strip()

        return im, caption


if __name__ == "__main__":
    dataset =  StyleDataset(data_path='data/rayonismDataset')
    print(len(dataset))
    # print(dataset[0])
    print(dataset[0][0].shape, dataset[0][1])
