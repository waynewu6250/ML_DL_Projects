import torch as t
import torch.nn as nn
from torch.autograd import Variable

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder):

        super(Seq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, inputs, targets):
        encoder_outputs, encoder_hidden = self.encoder.forward(inputs)
        decoder_outputs, decoder_hidden = self.decoder.forward(encoder_hidden, targets)
        return decoder_outputs, decoder_hidden
    
    def evaluation(self, inputs, decoder_hidden):
        _, encoder_hidden = self.encoder(inputs, decoder_hidden)
        decoded_sentence, decoder_hidden = self.decoder.evaluation(encoder_hidden)
        return decoded_sentence, decoder_hidden


