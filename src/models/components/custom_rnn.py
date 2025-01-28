import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    """
    A custom implementation of a Recurrent Neural Network (RNN) layer.

    This class supports various options for processing sequences, including:
    - Maintaining hidden states across batches (stateful RNN).
    - Returning the full sequence of outputs or just the final output.
    - Optionally returning the final hidden state.

    Attributes:
        cell (nn.Module): The RNN cell used for computation.
        return_sequences (bool): If True, returns the full sequence of outputs. Otherwise, returns the last output.
        return_state (bool): If True, returns the final hidden state along with the output(s).
        stateful (bool): If True, retains the hidden state between batches.
        states (torch.Tensor): Stores the hidden state if `stateful` is True.
    """
    def __init__(
        self,
        cell,  # A cell in RNN (e.g., LSTM, GRU, or custom RNN cell)
        return_sequences=False,  # Whether to return the entire sequence of outputs
        return_state=True,  # Whether to return the final state of the RNN
        stateful=False,  # Whether to maintain states across batches
        hidden_size=None,  # Hidden size of the cell (required if initial state is not provided)
        **kwargs,  # Additional arguments (not used here but allows flexibility)
    ):
        super(SimpleRNN, self).__init__()
        self.cell = cell  # The RNN cell used for computations (e.g., LSTMCell or a custom cell)
        self.return_sequences = return_sequences  # Whether to return the entire sequence of outputs
        self.return_state = return_state  # Whether to return the final state along with outputs
        self.stateful = stateful  # Whether to keep the state between batches
        self.hidden_size = hidden_size  # The hidden size of the cell
        self.states = None  # To store the state if stateful=True

    def forward(self, inputs, initial_state=None):
        """
        Forward pass for the RNN layer.

        Args:
            inputs: A tensor of shape (batch_size, timesteps, input_size).
            initial_state: Initial hidden state (optional). If None, the state will be initialized to zeros.

        Returns:
            Either the last output or the entire sequence, with the final state if specified.
        """
        batch_size, timesteps, *_ = inputs.size()  # Extract batch size and sequence length
        state = initial_state  # Initialize state

        # If stateful, use the stored state from the previous batch
        if self.stateful and self.states is not None:
            state = self.states

        # If no initial state is provided, initialize it to zeros
        elif state is None:            
            assert self.hidden_size is not None, "When `initial_state` is None, the RNN should have been initialized with the hidden size of the cell to create a valid initial state."
            state = torch.zeros((batch_size, self.hidden_size), device=inputs.device)

        outputs = []  # To store the outputs at each time step

        # Iterate through each time step in the input sequence
        for t in range(timesteps):
            # Pass the current input and state through the RNN cell
            output, state_and_output = self.cell(inputs[:, t, :], state)
            
            # Store the output of the current time step
            outputs.append(output)

        # If stateful, save the final state for the next batch
        if self.stateful:
            self.states = state_and_output

        # Stack the outputs along the time dimension to form the output sequence
        outputs = torch.stack(outputs, dim=1)

        # Determine what to return based on return_sequences and return_state
        if self.return_sequences:
            result = outputs  # Return the entire sequence of outputs
        else:
            result = outputs[:, -1, :]  # Return only the last output (last timestep)

        if self.return_state:
            return result, state_and_output  # Return the result and final state

        return result  # Return the result (outputs or last output)
