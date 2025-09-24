
#!/usr/bin/env python3
"""
Retail Shelf Image Captioning CLI Tool
Usage: python caption.py image.jpg
"""

import torch
import argparse
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import json
import time

from src.modeling.caption_model import RetailShelfCaptioner, LightweightCaptioner

class RetailShelfCaptioner_CLI:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load model checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)

        # Load vocabulary
        self.word_to_idx = checkpoint['word_to_idx']
        self.idx_to_word = checkpoint['idx_to_word']
        self.vocab_size = checkpoint['vocab_size']

        # Initialize model
        self.model = RetailShelfCaptioner(self.vocab_size)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])

        print(f"Model loaded successfully on {self.device}")

    def generate_caption(self, image_path):
        """Generate caption for a single image"""
        start_time = time.time()

        # Load and preprocess image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            return f"Error loading image: {e}"

        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Generate caption
        with torch.no_grad():
            generated_tokens = self.model(image_tensor)

        # Convert tokens to text
        caption_words = []
        for token_idx in generated_tokens[0]:
            word = self.idx_to_word[token_idx.item()]
            if word == '<END>':
                break
            if word not in ['<START>', '<PAD>', '<UNK>']:
                caption_words.append(word)

        caption = ' '.join(caption_words)

        # Ensure caption is under 12 words
        words = caption.split()
        if len(words) > 12:
            caption = ' '.join(words[:12])

        inference_time = time.time() - start_time

        return caption, inference_time

def main():
    parser = argparse.ArgumentParser(description='Generate captions for retail shelf images')
    parser.add_argument('image_path', type=str, help='Path to input image')
    parser.add_argument('--model_path', type=str, default='./models/checkpoints/best_model.pth',
                       help='Path to trained model')
    parser.add_argument('--verbose', action='store_true', help='Show timing information')

    args = parser.parse_args()

    # Check if image exists
    if not Path(args.image_path).exists():
        print(f"Error: Image file '{args.image_path}' not found")
        return

    # Check if model exists
    if not Path(args.model_path).exists():
        print(f"Error: Model file '{args.model_path}' not found")
        print("Please train the model first using: python src/train_model.py")
        return

    # Initialize captioner
    try:
        captioner = RetailShelfCaptioner_CLI(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Generate caption
    try:
        caption, inference_time = captioner.generate_caption(args.image_path)

        # Output caption
        print(caption)

        if args.verbose:
            print(f"Inference time: {inference_time:.3f} seconds")

    except Exception as e:
        print(f"Error generating caption: {e}")

if __name__ == "__main__":
    main()
