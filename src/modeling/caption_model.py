import ssl
import torch
import torch.nn as nn
import torchvision.models as models

# Temporary SSL fix for downloading pretrained models
ssl._create_default_https_context = ssl._create_unverified_context


class RetailShelfCaptioner(nn.Module):
    """Image captioning model for retail shelves"""

    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2):
        super(RetailShelfCaptioner, self).__init__()

        # Visual encoder (pretrained ResNet)
        resnet = models.resnet50(pretrained=True)
        # Remove the final classification layer
        self.encoder = nn.Sequential(*list(resnet.children())[:-1])

        # Freeze early layers (optional - for faster training)
        for param in list(self.encoder.parameters())[:-10]:
            param.requires_grad = False

        # Project visual features
        self.feature_projection = nn.Linear(2048, embed_dim)

        # Caption decoder
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)
        self.dropout = nn.Dropout(0.3)

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

    def forward(self, images, captions=None):
        batch_size = images.size(0)

        # Encode images
        with torch.no_grad():
            visual_features = self.encoder(images)  # (batch_size, 2048, 1, 1)

        visual_features = visual_features.view(batch_size, -1)  # (batch_size, 2048)
        visual_features = self.feature_projection(visual_features)  # (batch_size, embed_dim)

        if captions is not None:
            # Training mode
            caption_embeddings = self.embedding(captions)  # (batch_size, seq_len, embed_dim)

            # Concatenate visual features as first token
            visual_features = visual_features.unsqueeze(1)  # (batch_size, 1, embed_dim)
            inputs = torch.cat([visual_features, caption_embeddings[:, :-1]], dim=1)

            # LSTM forward pass
            lstm_out, _ = self.lstm(inputs)
            outputs = self.output_projection(self.dropout(lstm_out))

            return outputs
        else:
            # Inference mode - generate captions
            return self.generate_caption(visual_features)

    def generate_caption(self, visual_features, max_length=12, start_token=1, end_token=2):
        """Generate caption using greedy decoding"""
        batch_size = visual_features.size(0)

        # Initialize with visual features
        hidden = None
        inputs = visual_features.unsqueeze(1)  # (batch_size, 1, embed_dim)

        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=visual_features.device)

        for _ in range(max_length - 1):
            # LSTM forward pass
            lstm_out, hidden = self.lstm(inputs, hidden)
            outputs = self.output_projection(lstm_out)

            # Get next token
            next_token = outputs.argmax(dim=-1)  # (batch_size, 1)
            generated = torch.cat([generated, next_token], dim=1)

            # Prepare next input
            inputs = self.embedding(next_token)

            # Stop if all sequences generated end token
            if (next_token == end_token).all():
                break

        return generated

class LightweightCaptioner(nn.Module):
    """Lighter version using ViT + smaller decoder"""

    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256):
        super(LightweightCaptioner, self).__init__()

        # Use smaller visual encoder
        from torchvision.models import mobilenet_v3_small
        mobilenet = mobilenet_v3_small(pretrained=True)
        self.encoder = nn.Sequential(*list(mobilenet.children())[:-1])

        # Smaller projections
        self.feature_projection = nn.Linear(576, embed_dim)  # MobileNet output
        self.embedding = nn.Embedding(vocab_size, embed_dim)

        # Use GRU instead of LSTM (faster)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        self.vocab_size = vocab_size

    def forward(self, images, captions=None):
        batch_size = images.size(0)

        # Encode images
        visual_features = self.encoder(images)
        visual_features = visual_features.view(batch_size, -1)
        visual_features = self.feature_projection(visual_features)

        if captions is not None:
            # Training mode
            caption_embeddings = self.embedding(captions)
            visual_features = visual_features.unsqueeze(1)
            inputs = torch.cat([visual_features, caption_embeddings[:, :-1]], dim=1)

            gru_out, _ = self.gru(inputs)
            outputs = self.output_projection(gru_out)
            return outputs
        else:
            # Inference mode
            return self.generate_caption(visual_features)

    def generate_caption(self, visual_features, max_length=12, start_token=1, end_token=2):
        """Fast caption generation"""
        batch_size = visual_features.size(0)
        hidden = None
        inputs = visual_features.unsqueeze(1)

        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=visual_features.device)

        for _ in range(max_length - 1):
            gru_out, hidden = self.gru(inputs, hidden)
            outputs = self.output_projection(gru_out)
            next_token = outputs.argmax(dim=-1)
            generated = torch.cat([generated, next_token], dim=1)
            inputs = self.embedding(next_token)

            if (next_token == end_token).all():
                break

        return generated
