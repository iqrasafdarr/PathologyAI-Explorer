"""
Train/eval transform pipelines for PCam patches.

Design notes:
- ResNet50 here is ImageNet-pretrained, so inputs are normalized with
  ImageNet's channel mean/std. This matches the statistics the pretrained
  weights were trained on, which is what makes transfer learning work well —
  using different normalization would feed the network out-of-distribution
  inputs relative to its pretrained filters.
- Training augmentation is deliberately mild. PCam labels depend on the
  presence of tumor tissue in the *center* 32x32px region of each 96x96
  patch — aggressive geometric augmentation (e.g. large crops, heavy
  perspective warps) risks moving that center region or destroying the fine
  tissue structure the classifier depends on. Flips and small rotations
  preserve the label-relevant center region and reflect real rotational/
  mirror symmetry in histopathology slides (there's no canonical
  "orientation" for a tissue patch). Color jitter is kept mild to simulate
  realistic stain variation without erasing tissue morphology.
"""

from __future__ import annotations

import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def get_train_transforms(image_size: int = 96) -> T.Compose:
    """Training transforms: mild augmentation + ImageNet normalization."""
    return T.Compose(
        [
            T.ToPILImage(),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.5),
            T.RandomRotation(degrees=10),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_eval_transforms(image_size: int = 96) -> T.Compose:
    """Evaluation transforms: no augmentation, only resize + normalization."""
    return T.Compose(
        [
            T.ToPILImage(),
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
