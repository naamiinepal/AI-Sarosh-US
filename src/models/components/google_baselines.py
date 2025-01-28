import torch
from torch import nn
from torchvision.models import MobileNetV2
from grouped_conv_lstm_2d import GroupedConvLSTMCell
from custom_rnn import SimpleRNN
import torch.nn.functional as F

#Implementations based on  https://github.com/Google-Health/google-health/blob/master/fetal_ultrasound_blind_sweeps/networks.py

class FetalPresentationModel(nn.Module):
    """
    A PyTorch implementation of a fetal presentation regression model.

    This model predicts fetal presentation based on input video clips. 
    Args:
        base_network (nn.Module): The base network used to extract features from input video clips.
        feature_dim (int): The dimensionality of the spatially averaged features.
        is_training (bool): Whether the model is in training mode.
        num_classes (int): Number of output classes for classification. Defaults to 2.
    """
    def __init__(self, base_network, feature_dim, is_training, num_classes=2):
        super(FetalPresentationModel, self).__init__()
        self.base_network = base_network  # Base feature extractor network (e.g., LSTM, CNN, etc.)
        
        # Fully connected layer and activation function based on the number of classes
        if num_classes == 2:
            self.fetal_presentation = nn.Linear(feature_dim, 1)  # Single output for binary classification
            self.activation = nn.Sigmoid()  # Sigmoid activation for binary classification
        else:
            self.fetal_presentation = nn.Linear(feature_dim, num_classes)  # Multiple outputs for multi-class classification
            self.activation = nn.Softmax(dim=1)  # Softmax activation for multi-class classification
        
        self.training = is_training  # Flag to indicate training mode

    def forward(self, video_clips):
        """
        Forward pass for the fetal presentation classification model.

        Args:
            video_clips (torch.Tensor): Input tensor containing video clips. 
                Shape: (batch_size, seq_len, feature_dim).

        Returns:
            torch.Tensor: Predicted logits or probabilities depending on the mode.
        """
        # Extract features using the base network
        lstm_state_and_output = self.base_network(video_clips)
        
        # Spatially average features over all dimensions except batch and feature
        spatially_averaged_features = torch.mean(lstm_state_and_output, dim=(-1, -2))
        
        # Compute logits using the fully connected layer
        logits = self.fetal_presentation(spatially_averaged_features)
        print(logits.shape)
        # Apply activation function if in training mode
        if self.training:
            return self.activation(logits)
        return logits
       
        
    
class GARegressionModel(nn.Module):
    """
    A PyTorch implementation of a gestational age regression model.

    This model predicts gestational age and its variance based on input video clips. 
    The base network (e.g., an LSTM or another feature extractor) processes the input video 
    clips to extract features, which are then spatially averaged. Fully connected layers 
    predict the age and variance outputs. A softplus activation ensures the variance is 
    always positive, and a small positive constant is added to prevent numerical instability.

    Args:
        base_network (nn.Module): The base network used to extract features from input video clips.
        feature_dim (int): The dimensionality of the spatially averaged features.
    """
    def __init__(self, base_network, feature_dim):
        """
        Initializes the GARegressionModel.

        Args:
            base_network (nn.Module): The base feature extraction network.
            feature_dim (int): Dimensionality of the features after spatial averaging.
        """
        super(GARegressionModel, self).__init__()

        # Base feature extractor network (e.g., LSTM, CNN, etc.)
        self.base_network = base_network

        # Fully connected layer to predict gestational age
        self.age_output_layer = nn.Linear(feature_dim, 1)

        # Fully connected layer to predict variance of gestational age
        self.variance_output_layer = nn.Linear(feature_dim, 1)

    def forward(self, video_clips):
        """
        Forward pass for the gestational age regression model.

        Args:
            video_clips (torch.Tensor): Input tensor containing video clips. 
                Shape: (batch_size, seq_len, feature_dim).
            is_training (bool): Flag indicating whether the model is in training mode.

        Returns:
            age_output (torch.Tensor): Predicted gestational age. Shape: (batch_size, 1).
            variance_output (torch.Tensor): Predicted variance of gestational age. Shape: (batch_size, 1).
        """
        # Pass input through the base network to extract features
        lstm_state_and_output = self.base_network(video_clips)
        
        spatially_averaged_features = torch.mean(lstm_state_and_output, dim=(-1,-2))

       
        # Compute age predictions using a fully connected layer
        age_output = self.age_output_layer(spatially_averaged_features)

        # Compute variance predictions using a fully connected layer
        variance_output = self.variance_output_layer(spatially_averaged_features)

        # Apply softplus activation to ensure variance is positive, and add a small constant
        variance_output = 1e-6 + F.softplus(variance_output)

        return age_output, variance_output

class BaseGoogle(nn.Module):
   
    def __init__(
        self,
        extractor: nn.Module = MobileNetV2(),
        seq_processor: nn.Module = SimpleRNN(GroupedConvLSTMCell(filter_size=3, spatial_output_size=(7, 7), num_units=512))
    ) -> None:
        """
        Initialize the neural network with configurable architectures.

        Args:
            feat_extractor (nn.Module): Feature extractor module (default: FeatureExtractor).
            input_channels (int): Number of input channels for the backbone. Default is 1 (grayscale input).
            backbone_features (int): Number of features extracted by the backbone. Default is 512.
            classifier_conv_features (List[int]): Sizes for convolutional classifier layers.
            classifier_linear_sizes (List[int]): Sizes for linear classifier layers.
            dropout_rate (float): Dropout rate for regularization. Default is 0.4.
        """
        super().__init__()
        
        extractor_layers: list[nn.Module] = list(extractor.children())[:-1]
        self.extractor = nn.Sequential(*extractor_layers)

        self.seq_processor = seq_processor


    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, width, height).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_features).
        """
        batch_size, seq_len, channels, width, height = videos.size()

        # Reshape and process input through the feature extractor
        x_reshaped = videos.view(batch_size * seq_len, channels, width, height).float()
        feature_map = self.extractor(x_reshaped)
        # Rearrange features for further processing
        _, channels, width, height = feature_map.size()
        feature_seq = feature_map.view(batch_size, seq_len, channels, width, height)

        #check
        initial_state = self.seq_processor.cell.init_state(batch_size) 
        _, state_and_output = self.seq_processor(feature_seq, initial_state)

        state_and_output_concat = torch.concat(state_and_output, dim=1)
        
        return state_and_output_concat
    


