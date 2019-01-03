import random
import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

class AttnDecoder(nn.Module):
    def __init__(self, num_decoder_tokens, char_dim, latent_dim, time_steps, teacher_forcing_ratio, sos_id, dropout_rate=0.1):
        """Define layers for attention decoder"""
        super(AttnDecoder, self).__init__()

        self.embedding = nn.Embedding(num_decoder_tokens, char_dim)
        self.lstm = nn.LSTM(char_dim, latent_dim)
        self.out = nn.Linear(latent_dim, num_decoder_tokens)
        self.log_softmax = nn.LogSoftmax(dim=1)  # work with NLLLoss = CrossEntropyLoss

        # Attention Layer
        self.attn = nn.Linear(latent_dim+char_dim, time_steps)
        self.attn_combine = nn.Linear(latent_dim*2, latent_dim)
        self.dropout = nn.Dropout(dropout_rate)

        self.latent_dim = latent_dim
        self.char_dim = char_dim
        self.time_steps = time_steps
        self.num_decoder_tokens = num_decoder_tokens
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.sos_id = sos_id
        self.dropout_rate = dropout_rate
    
    def forward_step(self, decoder_input, decoder_hidden, encoder_outputs):
        batch_size = decoder_input.size(1)
        outputs = encoder_outputs.view(batch_size, self.time_steps, -1)

        # inputs: (time_steps=1, batch_size)
        embedded = self.embedding(decoder_input).view(1,1,-1)
       
        #===========Attention=============#
         # embedded: (1, batch_size, char_dim)
        embedded = self.dropout(embedded).view(1,-1,self.char_dim)

        # concat it with the hidden layer and send it into attention and softmax layer
        attn_weights = F.softmax(self.attn(torch.cat((embedded[0], decoder_hidden[0][0]), 1)), dim=1)

        attn_applied = torch.bmm(attn_weights.unsqueeze(1), outputs)
        attn_applied = attn_applied.view(1,batch_size,-1)
        output = F.relu(self.attn_combine(torch.cat((embedded[0], attn_applied[0]),1)).unsqueeze(0))
        #===========Attention=============#

        rnn_output, hidden = self.lstm(output, decoder_hidden)  # (1, batch_size, latent_dim)
        rnn_output = rnn_output.squeeze(0)  # squeeze the time dimension (batch_size, latent_dim)
        output = self.log_softmax(self.out(rnn_output))  # (batch_size, vocab_size)
        return output, hidden

    def forward(self, encoder_outputs, context_vector, targets):

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
            decoder_outputs_on_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
            decoder_outputs[t] = decoder_outputs_on_t
            if use_teacher_forcing:
                # Correct answer
                decoder_input = targets[t].unsqueeze(0)
            else:
                # Its own last output
                _, index = torch.topk(decoder_outputs_on_t, 1)
                decoder_input = index.transpose(0, 1)
        
        return decoder_outputs, decoder_hidden
    
    def evaluation(self, encoder_outputs, decoder_hidden):

        batch_size = decoder_hidden[0].size(1)
        decoder_input = Variable(torch.LongTensor([[self.sos_id] * batch_size]))
        decoder_outputs = Variable(torch.zeros(
            20,
            batch_size,
            self.num_decoder_tokens
        ))  # (time_steps, batch_size, vocab_size)

        # Unfold the decoder RNN on the time dimension
        for t in range(20):
            decoder_outputs_on_t, decoder_hidden = self.forward_step(decoder_input, decoder_hidden, encoder_outputs)
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
            


