import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import argparse
from pathlib import Path
import json
from tqdm import tqdm
import time

from preprocessing.data_loader import create_data_loaders
from modeling.caption_model import RetailShelfCaptioner, LightweightCaptioner

def train_model(args):
    """Main training function"""

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Create data loaders
    train_loader, val_loader, test_loader, vocab_size, word_to_idx, idx_to_word = create_data_loaders(
        args.data_dir, batch_size=args.batch_size
    )

    print(f"Dataset loaded: {len(train_loader.dataset)} train, {len(val_loader.dataset)} val, {len(test_loader.dataset)} test")
    print(f"Vocabulary size: {vocab_size}")

    # Create model
    if args.model_type == 'lightweight':
        model = LightweightCaptioner(vocab_size)
    else:
        model = RetailShelfCaptioner(vocab_size)

    model = model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(ignore_index=word_to_idx['<PAD>'])
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.8)

    # Tensorboard logging
    writer = SummaryWriter(f'runs/retail_captioning_{int(time.time())}')

    # Training loop
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        # Training
        model.train()
        train_loss = 0
        train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')

        for batch_idx, batch in enumerate(train_pbar):
            images = batch['image'].to(device)
            captions = batch['caption'].to(device)

            optimizer.zero_grad()

            # Forward pass
            outputs = model(images, captions)

            # Calculate loss
            loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()
            train_pbar.set_postfix({'loss': loss.item()})

            # Log to tensorboard
            if batch_idx % 10 == 0:
                writer.add_scalar('Loss/Train_Step', loss.item(), epoch * len(train_loader) + batch_idx)

        # Validation
        model.eval()
        val_loss = 0
        val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]')

        with torch.no_grad():
            for batch in val_pbar:
                images = batch['image'].to(device)
                captions = batch['caption'].to(device)

                outputs = model(images, captions)
                loss = criterion(outputs.view(-1, vocab_size), captions.view(-1))
                val_loss += loss.item()
                val_pbar.set_postfix({'loss': loss.item()})

        # Calculate average losses
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)

        # Log to tensorboard
        writer.add_scalar('Loss/Train_Epoch', avg_train_loss, epoch)
        writer.add_scalar('Loss/Val_Epoch', avg_val_loss, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')

        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
                'vocab_size': vocab_size,
                'word_to_idx': word_to_idx,
                'idx_to_word': idx_to_word
            }, args.save_path / 'best_model.pth')
            print(f'New best model saved with val loss: {avg_val_loss:.4f}')

        scheduler.step()

        # Generate sample captions every 5 epochs
        if (epoch + 1) % 5 == 0:
            generate_sample_captions(model, val_loader, idx_to_word, device, num_samples=3)

    writer.close()
    print("Training completed!")

def generate_sample_captions(model, data_loader, idx_to_word, device, num_samples=3):
    """Generate sample captions during training"""
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            if i >= num_samples:
                break

            images = batch['image'].to(device)
            true_captions = batch['caption_text']

            # Generate captions
            generated_tokens = model(images[:1])  # Generate for first image only

            # Convert tokens to text
            generated_caption = []
            for token_idx in generated_tokens[0]:
                word = idx_to_word[token_idx.item()]
                if word == '<END>':
                    break
                if word not in ['<START>', '<PAD>']:
                    generated_caption.append(word)

            print(f"Sample {i+1}:")
            print(f"  True: {true_captions[0]}")
            print(f"  Generated: {' '.join(generated_caption)}")
            print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Retail Shelf Captioning Model')
    parser.add_argument('--data_dir', type=str, default='./data', help='Path to dataset')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--model_type', type=str, default='standard', choices=['standard', 'lightweight'])
    parser.add_argument('--save_path', type=str, default='./models/checkpoints', help='Model save path')

    args = parser.parse_args()
    args.save_path = Path(args.save_path)
    args.save_path.mkdir(parents=True, exist_ok=True)

    train_model(args)
