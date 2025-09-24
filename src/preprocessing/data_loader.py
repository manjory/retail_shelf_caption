import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import json
from pathlib import Path
import random

class RetailShelfDataset(Dataset):
    """Custom dataset for retail shelf images with captions"""

    def __init__(self, data_dir, split='train', transform=None, max_caption_length=12):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        self.max_caption_length = max_caption_length

        # Load dataset splits
        with open(self.data_dir / 'dataset_splits.json', 'r') as f:
            splits = json.load(f)

        # Load captions
        with open(self.data_dir / 'raw' / 'captions.json', 'r') as f:
            self.captions = json.load(f)

        # Get image names for this split
        self.image_names = splits[split]

        # Build vocabulary
        self.build_vocabulary()

    def build_vocabulary(self):
        """Build vocabulary from all captions"""
        vocab = set(['<PAD>', '<START>', '<END>', '<UNK>'])

        for caption in self.captions.values():
            words = caption.lower().split()
            vocab.update(words)

        self.vocab = sorted(list(vocab))
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.vocab)

    def caption_to_tokens(self, caption):
        """Convert caption to token indices"""
        words = ['<START>'] + caption.lower().split()[:self.max_caption_length-2] + ['<END>']
        tokens = [self.word_to_idx.get(word, self.word_to_idx['<UNK>']) for word in words]

        # Pad to max length
        while len(tokens) < self.max_caption_length:
            tokens.append(self.word_to_idx['<PAD>'])

        return torch.tensor(tokens[:self.max_caption_length])

    def __len__(self):
        return len(self.image_names)

    def __getitem__(self, idx):
        image_name = self.image_names[idx]

        # Load image
        image_path = self.data_dir / 'raw' / 'images' / image_name
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        # Get caption
        caption = self.captions[image_name]
        caption_tokens = self.caption_to_tokens(caption)

        return {
            'image': image,
            'caption': caption_tokens,
            'caption_text': caption,
            'image_name': image_name
        }

def get_transforms(split='train'):
    """Get image transforms for training/validation"""

    if split == 'train':
        # Data augmentation for training
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(0.3),
            transforms.RandomRotation(5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        # No augmentation for validation/test
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

def create_data_loaders(data_dir, batch_size=16):
    """Create data loaders for train/val/test"""

    # Create datasets
    train_dataset = RetailShelfDataset(
        data_dir, split='train',
        transform=get_transforms('train')
    )

    val_dataset = RetailShelfDataset(
        data_dir, split='validation',
        transform=get_transforms('val')
    )

    test_dataset = RetailShelfDataset(
        data_dir, split='test',
        transform=get_transforms('test')
    )

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, train_dataset.vocab_size, train_dataset.word_to_idx, train_dataset.idx_to_word
