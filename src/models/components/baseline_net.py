import torch
import torch.nn as nn
from torchvision.models import resnet18
from typing import List
from feat_extractor import FeatureExtractor


class SimpleBaseNet(nn.Module):
    """
    A neural network utilizing a feature extractor (ResNet18 by default) as a backbone
    and custom convolutional and linear layers for classification.
    """

    def __init__(
        self,
        feat_extractor: nn.Module = FeatureExtractor(),
        classifier_conv_features: List[int] = [512, 256, 64, 1],
        classifier_linear_sizes: List[int] = [114, 64, 4],
        dropout_rate: float = 0.4,
    ) -> None:
        """
        Initialize the neural network with configurable architectures.

        Args:
            feat_extractor (nn.Module): Feature extractor module (default: FeatureExtractor).
            classifier_conv_features (List[int]): Sizes for convolutional classifier layers.
            classifier_linear_sizes (List[int]): Sizes for linear classifier layers.
            dropout_rate (float): Dropout rate for regularization. Default is 0.4.
        """
        super().__init__()

        self.extractor = feat_extractor
        self.dropout = nn.Dropout(dropout_rate)

        # Convolutional classifier
        self.classifier_conv = nn.Sequential(
            nn.AdaptiveAvgPool1d(128),  # Fixed-size pooling
            *self._create_conv_block(classifier_conv_features),
        )

        # Linear classifier
        self.classifier_lin = nn.Sequential(
            *self._create_linear_block(classifier_linear_sizes),
        )

    def _create_conv_block(self, feature_sizes: List[int]) -> List[nn.Module]:
        """
        Create a sequence of 1D convolutional layers with activation and normalization.

        Args:
            feature_sizes (List[int]): Sizes of convolutional layers.

        Returns:
            List[nn.Module]: List of layers for the convolutional block.
        """
        layers: List[nn.Module] = []
        for i in range(len(feature_sizes) - 1):
            layers.extend([
                nn.Conv1d(feature_sizes[i], feature_sizes[i + 1], kernel_size=7 - 2 * i),
                nn.ReLU(),
                nn.BatchNorm1d(feature_sizes[i + 1]),
            ])
        return layers

    def _create_linear_block(self, sizes: List[int]) -> List[nn.Module]:
        """
        Create a sequence of linear layers with activation.

        Args:
            sizes (List[int]): Sizes of linear layers.

        Returns:
            List[nn.Module]: List of layers for the linear block.
        """
        layers: List[nn.Module] = []
        for i in range(len(sizes) - 1):
            layers.extend([
                nn.Linear(sizes[i], sizes[i + 1]),
                nn.ReLU(),
            ])
        return layers[:-1]  # Remove the last ReLU for output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, width, height).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_features).
        """
        batch_size, seq_len, width, height = x.size()

        # Reshape and process input through the feature extractor
        x_reshaped = x.view(batch_size * seq_len, width, height).unsqueeze(dim=1).float()
        feature_map = self.extractor(x_reshaped, seq_len)

        # Rearrange features for further processing
        _, channels, width, height = feature_map.size()
        feature_seq = feature_map.view(batch_size, seq_len, channels, width, height)
        feature_seq = feature_seq.permute(0, 2, 3, 4, 1).reshape(batch_size, channels, width * height * seq_len)

        # Apply dropout
        feature_seq = self.dropout(feature_seq)

        # Convolutional classification
        conv_features = self.classifier_conv(feature_seq)
        conv_features = self.dropout(conv_features)

        # Linear classification
        linear_input = conv_features.view(batch_size, -1)
        output = self.classifier_lin(linear_input)

        return output


if __name__ == "__main__":
    # Example instantiation
    model = SimpleBaseNet()

    # Example input: (batch_size, seq_len, width, height)
    dummy_input = torch.randn(4, 10, 64, 64)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
