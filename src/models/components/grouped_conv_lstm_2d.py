import torch
import torch.nn as nn
from separable_conv import SeparableConv2D
from custom_rnn import SimpleRNN

class GroupedConvLSTMCell(nn.Module):
    """
    A grouped Convolutional LSTM (ConvLSTM) cell implementation, where convolutions are used
    for both input-to-state and state-to-state transitions. This is particularly useful for
    processing spatiotemporal data like video or image sequences. 
    Based on
    https://github.com/tensorflow/models/blob/master/research/lstm_object_detection/lstm/lstm_cells.py
    with a subset of functionalities.
    """
    
    def __init__(self, filter_size, spatial_output_size, num_units,
                 forget_bias=1.0, activation=torch.tanh, use_batch_norm=False,
                 groups=4, scale_state=False, clip_state=False,
                 visualize_gates=False):
        """
        Initializes the GroupedConvLSTMCell with the given hyperparameters.
        
        Args:
            filter_size (int or tuple): The size of the convolutional kernel.
            spatial_output_size (tuple): The spatial size of the output (height, width).
            num_units (int): The number of hidden units in the LSTM cell.
            forget_bias (float): The bias added to the forget gate. Default is 1.0.
            activation (callable): Activation function for the cell. Default is torch.tanh.
            use_batch_norm (bool): Whether to apply batch normalization. Default is False.
            groups (int): The number of groups for splitting channels. Default is 4.
            scale_state (bool): Whether to scale the state. Default is False.
            clip_state (bool): Whether to clip the state values. Default is False.
            visualize_gates (bool): Whether to visualize the gates. Default is False.
        """
        super(GroupedConvLSTMCell, self).__init__()
        self.filter_size = int(filter_size)
        self.spatial_output_size = list(spatial_output_size)
        self.num_units = num_units
        self.forget_bias = forget_bias
        self.activation = activation
        self.use_batch_norm = use_batch_norm
        self.viz_gates = visualize_gates
        self.groups = groups
        self.scale_state = scale_state
        self.clip_state = clip_state
        
    def forward(self, inputs, state):
        """
        Performs a forward pass through the GroupedConvLSTMCell.
        
        Args:
            inputs (Tensor): The input tensor, typically with shape (batch_size, channels, height, width).
            state (tuple): The current state of the LSTM, consisting of the cell state (c) and hidden state (h).
        
        Returns:
            output (Tensor): The output at the current time step.
            new_c (Tensor): The updated cell state.
            new_h (Tensor): The updated hidden state.
        """
        c, h = state
        
        # Split cell state and hidden state into groups
        c_list = torch.chunk(c, self.groups, dim=1)
        h_list = torch.chunk(h, self.groups, dim=1)
            
        batch_size_i, channels_i, height_i, width_i = inputs.shape 
        batch_size_h, channels_h, height_h, width_h = h_list[0].shape    
   
        out_c = []
        out_h = []
        
        for k in range(self.groups):
            bottleneck_cls = SeparableConv2D(kernel_size=self.filter_size, activation_fn=self.activation)
        
            # Optionally apply batch normalization
            if self.use_batch_norm:
                out_channels = self.num_units // self.groups
                b_x = bottleneck_cls(inputs, channels_i, out_channels, 1)
                b_h = bottleneck_cls(h_list[k], channels_h, out_channels, 1)
                b_x = nn.BatchNorm2d(out_channels)(b_x)
                b_h = nn.BatchNorm2d(out_channels)(b_h)
                bottleneck = b_x + b_h
            else:
                bottleneck_cat = torch.concat((inputs, h_list[k]), dim=1)
                channels_cat = bottleneck_cat.shape[1]
                bottleneck = bottleneck_cls(bottleneck_cat, channels_cat, self.num_units // self.groups, 1)
                    
            # Convolution to combine bottleneck with previous states
            conv_cls = SeparableConv2D(kernel_size=self.filter_size, activation_fn=None)
            concat = conv_cls(bottleneck, self.num_units // self.groups, 4 * self.num_units // self.groups, 1)
            
            # Split the convolution output into gates: input, forget, output, and candidate
            i, j, f, o = torch.chunk(concat, 4, dim=1)
            f_add = f + self.forget_bias
            f_act = torch.sigmoid(f_add)
            
            a = c_list[k] * f_act  # Update cell state
            i_act = torch.sigmoid(i)
            j_act = self.activation(j)
            b = i_act * j_act  # Compute candidate state
            
            new_c = a + b  # Update cell state
            
            o_act = torch.sigmoid(o)
            new_h = self.activation(new_c) * o_act  # Update hidden state
            
            # Append the results for this group
            out_c.append(new_c)
            out_h.append(new_h)  
            
        # Concatenate outputs from all groups
        new_c = torch.concat(out_c, axis=1)
        new_h = torch.concat(out_h, axis=1)
        
        # Return the final output, new cell state, and new hidden state
        output = new_h 
        return (output, (new_c, new_h))
    
    def init_state(self, batch_size, dtype=torch.float32):
        state_size = ([self.num_units]+ self.spatial_output_size  , [self.num_units]+ self.spatial_output_size )
        c = torch.zeros([batch_size] + state_size[0], dtype=dtype)
        h = torch.zeros([batch_size] + state_size[1], dtype=dtype)
        return c, h    
