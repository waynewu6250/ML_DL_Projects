import random
import torch
from torch import nn
from torch.autograd import Variable

class Decoder(nn.Module):
    def __init__(self, num_decoder_tokens, char_dim, latent_dim, teacher_forcing_ratio, sos_id):
        """Define layers for decoder"""
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(num_decoder_tokens, char_dim)
        self.lstm = nn.LSTM(char_dim, latent_dim)
        self.out = nn.Linear(latent_dim, num_decoder_tokens)
        self.log_softmax = nn.LogSoftmax(dim=1)  # work with NLLLoss = CrossEntropyLoss

        self.latent_dim = latent_dim
        self.char_dim = char_dim
        self.num_decoder_tokens = num_decoder_tokens
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.sos_id = sos_id
    
    def forward_step(self, inputs, hidden):
        # inputs: (time_steps=1, batch_size)
        batch_size = inputs.size(1)
        embedded = self.embedding(inputs) # (batch_size, char_dim)
        embedded.view(1, batch_size, self.char_dim)  # (1, batch_size, char_dim)
        rnn_output, hidden = self.lstm(embedded, hidden)  # (1, batch_size, latent_dim)
        rnn_output = rnn_output.squeeze(0)  # squeeze the time dimension (batch_size, latent_dim)
        output = self.log_softmax(self.out(rnn_output))  # (batch_size, vocab_size)
        return output, hidden
    
    def forward(self, context_vector, targets):

        # Prepare variable for decoder on time_step_0
        batch_size = context_vector[0].size(1)
        decoder_input = Variable(torch.LongTensor([[self.sos_id] * batch_size]))
        decoder_hidden = context_vector
        
        decoder_outputs = Variable(torch.zeros(
            targets.size(0),
            batch_size,
            self.num_decoder_tokens
        ))  # (time_steps, batch_size, vocab_size)
        use_teacher_forcing = True if random.random() > self.teacher_forcing_ratio else False

        # Unfold the decoder RNN on the time dimension
        for t in range(targets.size(0)):
            decoder_outputs_on_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs[t] = decoder_outputs_on_t
            if use_teacher_forcing:
                # Correct answer
                decoder_input = targets[t].unsqueeze(0)
            else:
                # Its own last output
                _, index = torch.topk(decoder_outputs_on_t, 1)
                decoder_input = index.transpose(0, 1)
        
        return decoder_outputs, decoder_hidden
    
    def evaluation(self, decoder_hidden):

        batch_size = decoder_hidden[0].size(1)
        decoder_input = Variable(torch.LongTensor([[self.sos_id] * batch_size]))
        decoder_outputs = Variable(torch.zeros(
            20,
            batch_size,
            self.num_decoder_tokens
        ))  # (time_steps, batch_size, vocab_size)

        # Unfold the decoder RNN on the time dimension
        for t in range(20):
            decoder_outputs_on_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs[t] = decoder_outputs_on_t
            # Its own last output
            _, index = torch.topk(decoder_outputs_on_t, 1)
            decoder_input = index.transpose(0, 1)
        
        #  Get the output sequence from decoder
        decoded_indices = []
        batch_size = decoder_outputs.size(1)
        decoder_outputs = decoder_outputs.transpose(0, 1)  # S = B x T x V

        for b in range(batch_size):
            for t in range(20):
                _, index = torch.topk(decoder_outputs[b][t], 1)
                decoded_indices.append(index.item())
        
        return decoded_indices





    