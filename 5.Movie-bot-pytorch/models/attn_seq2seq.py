import torch as t
import torch.nn as nn
from torch.autograd import Variable

class AttnSeq2seq(nn.Module):
    def __init__(self, encoder, decoder):

        super(AttnSeq2seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self, inputs, targets):
        encoder_outputs, encoder_hidden = self.encoder.forward(inputs)
        decoder_outputs, decoder_hidden = self.decoder.forward(encoder_outputs, encoder_hidden, targets)
        return decoder_outputs, decoder_hidden
    
    def evaluation(self, inputs):
        encoder_outputs, encoder_hidden = self.encoder(inputs)
        decoded_sentence = self.decoder.evaluation(encoder_outputs, encoder_hidden)
        return decoded_sentence