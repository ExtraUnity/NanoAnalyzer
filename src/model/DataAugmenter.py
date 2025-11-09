from torch.utils.data import Dataset
from src.model.SegmentationDataset import RepeatDataset, SegmentationDataset
class DataAugmenter():
    def __init__(self):
        return 
    
    @staticmethod
    def get_transformer(crop: bool, rotate: bool, flip: bool, deform: bool, adjust_brightness: bool, adjust_contrast: bool, blur: bool):
        def transformer(image, mask):
            from torchvision.transforms.v2 import ElasticTransform, RandomCrop, GaussianBlur
            import torchvision.transforms.functional as TF
            from torchvision.transforms import InterpolationMode
            import random

            # Elastic deformation
            if deform:
                params = ElasticTransform.get_params(
                    size=[512, 512],
                    alpha=(20.0, 60),
                    sigma=(4.0, 6.0)
                )
                image = TF.elastic_transform(image, params)
                mask = TF.elastic_transform(mask, params, interpolation=InterpolationMode.NEAREST)

            # Random crop
            if crop:
                i, j, h, w = RandomCrop.get_params(
                    image, output_size=(256, 256))
                image = TF.crop(image, i, j, h, w)
                mask = TF.crop(mask, i, j, h, w)

            # Random rotation
            if rotate:
                angle = random.choice([0, 90, 180, 270])
                image = TF.rotate(image, angle)
                mask = TF.rotate(mask, angle)

            # Random horizontal flipping
            if flip and random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # Brightness adjustment
            if adjust_brightness:
                brightness_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_brightness(image, brightness_factor)

            # Contrast adjustment
            if adjust_contrast:
                contrast_factor = random.uniform(0.8, 1.2)
                image = TF.adjust_contrast(image, contrast_factor)

            # Random blur
            if blur:
                blur_transform = GaussianBlur(kernel_size=3, sigma=(0.5, 1.5))
                if random.random() < 0.5:
                    image = blur_transform(image)

            return image, mask

        return transformer

    
    def augment_dataset(self, dataset: Dataset, augmentations=[True,True,False,False,True,True,False]) -> Dataset:
        new_images = []
        new_masks = []
        new_filenames = []
        for i in range(len(dataset)):
            image, mask = dataset[i]
            # Handle both regular datasets and Subset datasets
            if hasattr(dataset, 'indices'):
                # This is a Subset, get the filename from the original dataset
                original_idx = dataset.indices[i]
                filename = dataset.dataset.image_filenames[original_idx]
            else:
                # This is a regular dataset
                filename = dataset.image_filenames[i]
            
            new_images.extend(image.unsqueeze(0))
            new_masks.extend(mask.unsqueeze(0))
            new_filenames.append(filename)
        
        return RepeatDataset(dataset=SegmentationDataset.from_image_set(new_images, new_masks, new_filenames, transforms=DataAugmenter.get_transformer(*augmentations)), repeat_factor=10 if augmentations[0] else 20)
    