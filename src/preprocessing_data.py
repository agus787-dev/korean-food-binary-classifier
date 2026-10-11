from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from utils import IMAGE_SIZE, MEAN, STD, TEST_DIR, TRAIN_DIR, VAL_DIR


def get_train_transforms():
    """
    Training transforms:
    1. resize all images to a fixed size
    2. apply small augmentation only for training
    3. convert image to tensor with shape [C, H, W]
    4. normalize pixel values
    """
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def get_eval_transforms():
    """Validation and test transforms without augmentation."""
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=MEAN, std=STD),
    ])


def create_imagefolder_dataset(data_dir: Path, transform):
    """Create an ImageFolder dataset from a split directory."""
    return datasets.ImageFolder(root=data_dir, transform=transform)


def create_dataloaders(batch_size: int = 16):
    """
    Create train, validation, and test DataLoaders.
    ImageFolder automatically uses subfolder names as labels.
    """
    train_dataset = create_imagefolder_dataset(TRAIN_DIR, get_train_transforms())
    val_dataset = create_imagefolder_dataset(VAL_DIR, get_eval_transforms())
    test_dataset = create_imagefolder_dataset(TEST_DIR, get_eval_transforms())

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    class_names = train_dataset.classes

    sample_images, sample_labels = next(iter(train_loader))
    print(f"Sample batch image shape: {sample_images.shape}")
    print(f"Sample batch label shape: {sample_labels.shape}")
    print("Class names:", class_names)

    return train_loader, val_loader, test_loader, class_names