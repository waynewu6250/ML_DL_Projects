import torch as t
from torch import nn

class Encoder(nn.Module):

    def __init__(self, num_encoder_tokens, char_dim, latent_dim):
        """Define layers for encoder"""
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(num_encoder_tokens, char_dim)
        self.lstm = nn.LSTM(char_dim, latent_dim)
    
    def forward(self, input_var, hidden=None):
        """
        input_var: (time_steps, batch_size)
        embedded: (time_steps, batch_size, char_dim)
        outputs, hidden: (time_steps, batch_size, latent_dim)
        """
        embedded = self.embedding(input_var)
        outputs, hidden = self.lstm(embedded, hidden)
        return outputs, hidden