import torch
import torch.nn as nn
from torchvision.models import resnet18
from typing import List


class FeatureExtractor(nn.Module):
    """
    A module for extracting features from input data using a configurable backbone network (ResNet18 by default).
    Supports optional chunking to process data in smaller batches for memory efficiency.
    """

    def __init__(
        self,
        input_channels: int = 1,
        backbone: nn.Module = resnet18(),
        enable_chunking: bool = True,
        chunk_size: int = 128,
    ) -> None:
        """
        Initialize the feature extractor.

        Args:
            input_channels (int): Number of input channels for the first convolution layer (default: 1 for grayscale input).
            backbone (nn.Module): Backbone model for feature extraction (default: ResNet18).
            enable_chunking (bool): Whether to process data in chunks for memory efficiency (default: True).
            chunk_size (int): Size of chunks to process when chunking is enabled (default: 128).
        """
        super().__init__()

        # Extract backbone layers, excluding the final classification layer
        backbone_layers: List[nn.Module] = list(backbone.children())[:-1]
        self.backbone = nn.Sequential(*backbone_layers)

        # Update the first convolution layer to match the specified input channels
        self.backbone[0] = nn.Conv2d(
            in_channels=input_channels,
            out_channels=64,
            kernel_size=(7, 7),
            stride=(2, 2),
            padding=(3, 3),
            bias=False,
        )

        self.enable_chunking = enable_chunking
        self.chunk_size = chunk_size

    def forward(self, x: torch.Tensor, seq_len: int) -> torch.Tensor:
        """
        Perform a forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor with shape (batch_size * seq_len, width, height).
            seq_len (int): Sequence length of each batch.

        Returns:
            torch.Tensor: Extracted feature map with shape (batch_size * seq_len, feature_size, 1, 1).
        """
        bseq, width, height = x.size()
        batch_size = bseq // seq_len  # Calculate the batch size based on input dimensions

        if self.enable_chunking:
            # Chunked processing for memory efficiency
            feature_maps: List[torch.Tensor] = []
            num_full_chunks = (batch_size * seq_len) // self.chunk_size
            num_remaining = (batch_size * seq_len) % self.chunk_size

            # Process full chunks
            for i in range(num_full_chunks):
                start_idx = i * self.chunk_size
                end_idx = start_idx + self.chunk_size
                chunk_output = self.backbone(x[start_idx:end_idx])
                feature_maps.append(chunk_output)

            # Process the remaining partial chunk
            if num_remaining > 0:
                remaining_output = self.backbone(x[-num_remaining:])
                feature_maps.append(remaining_output)

            # Combine all chunks into a single tensor
            feature_map = torch.cat(feature_maps, dim=0)
        else:
            # Process the entire input tensor at once
            feature_map = self.backbone(x)

        return feature_map