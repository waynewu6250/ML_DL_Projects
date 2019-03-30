import torch as t
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from eval_model.config import opt

class ClrModel(nn.Module):
    def __init__(self, word2id, num_classes, opt):
        super(ClrModel,self).__init__()
        self.embedding = nn.Embedding(len(word2id)+1, opt.embedding_dim)
        self.rnn = nn.LSTM(opt.embedding_dim, opt.hidden_dim, num_layers=opt.num_layers)
        self.fc = nn.Linear(opt.hidden_dim, num_classes)


    def forward(self, captions_t, lengths):
        embeddings = self.embedding(captions_t)
        pack_embeddings = pack_padded_sequence(embeddings, lengths)
        _, states = self.rnn(pack_embeddings)
        pred = self.fc(states[0])
        return pred

    def predict(self, captions_t):
        embeddings = self.embedding(captions_t)
        _, states = self.rnn(embeddings)
        pred = self.fc(states[0])
        return nn.Softmax(0)(pred)

