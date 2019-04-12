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
        rnn_output, hidden = self.lstm(embedded, hidden)  # (1, batch_size, latent_dim)
        rnn_output = rnn_output.squeeze(0)  # squeeze the time dimension (batch_size, latent_dim)
        output = self.out(rnn_output)  # (batch_size, vocab_size)
        return output, hidden
    
    def forward(self, decoder_hidden, targets):
        
        # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        # # Prepare variable for decoder on time_step_0
        # batch_size = decoder_hidden[0].size(1)
        # decoder_input = Variable(torch.LongTensor([[self.sos_id] * batch_size])).to(device)
        
        # decoder_outputs = Variable(torch.zeros(targets.size(0), batch_size, self.num_decoder_tokens)).to(device)  # (time_steps, batch_size, vocab_size)
        # use_teacher_forcing = True if random.random() > self.teacher_forcing_ratio else False

        # # Unfold the decoder RNN on the time dimension
        # for t in range(targets.size(0)):
        #     decoder_outputs_on_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
        #     decoder_outputs[t] = decoder_outputs_on_t
        #     if use_teacher_forcing:
        #         # Correct answer
        #         decoder_input = targets[t].unsqueeze(0)
        #     else:
        #         # Its own last output
        #         _, index = torch.topk(self.log_softmax(decoder_outputs_on_t), 1)
        #         decoder_input = index.transpose(0, 1)

        embedded = self.embedding(targets) # (time_steps, batch_size, char_dim)
        decoder_outputs, decoder_hidden = self.lstm(embedded, decoder_hidden)
        decoder_outputs = self.out(decoder_outputs)
        
        return decoder_outputs, decoder_hidden
    
    def evaluation(self, decoder_hidden):

        batch_size = decoder_hidden[0].size(1)
        decoder_input = Variable(torch.LongTensor([[self.sos_id] * batch_size]))
        decoder_outputs = Variable(torch.zeros(
            20,
            batch_size,
            self.num_decoder_tokens
        ))  # (time_steps, batch_size, vocab_size)
        decoded_indices = []

        # Unfold the decoder RNN on the time dimension
        for t in range(20):
            decoder_outputs_on_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs[t] = self.log_softmax(decoder_outputs_on_t)
            
            # Its own last output
            _, index = torch.topk(decoder_outputs[t], 1)
            decoder_input = index.transpose(0, 1)
            decoded_indices.append(index.item())
        
        return decoded_indices, decoder_hidden





    