import torch as t
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LyricsModel(nn.Module):

    def __init__(self, vocab_size, embedding_dim, latent_dim):
        
        super(LyricsModel, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.latent_dim = latent_dim

        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, latent_dim, num_layers=2)
        self.linear = nn.Linear(latent_dim, vocab_size)

    def forward(self, input, hidden=None):
        
        seq_len, batch_size = input.size()
        
        if hidden is None:
            h0 = Variable(input.data.new(2, batch_size, self.latent_dim).fill_(0).float())
            c0 = Variable(input.data.new(2, batch_size, self.latent_dim).fill_(0).float())
        else:
            h0,c0 = hidden
        
        embeddings = self.embeddings(input)
        output, hidden = self.lstm(embeddings, (h0,c0))
        output = self.linear(output.view(seq_len*batch_size,-1))
        return output,hidden
        

