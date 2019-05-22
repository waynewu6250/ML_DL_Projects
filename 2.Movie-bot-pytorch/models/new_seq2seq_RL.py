import torch
import torch.nn as nn
from torch.autograd import Variable
from models import NewSeq2seq
from config import opt
import random
import numpy as np

class NewSeq2seqRL(NewSeq2seq):
    def __init__(self, num_tokens, opt, sos_id):

        super(NewSeq2seqRL, self).__init__(num_tokens, opt, sos_id)
        self.criterion = nn.CrossEntropyLoss().to(self.device)
    
    def forward(self, inputs, targets, rewards):

        # Encoder Phase
        encoder_outputs, encoder_hidden1, encoder_hidden2 = self.encoder_forward(inputs)
        hidden1 = encoder_hidden1
        hidden2 = encoder_hidden2
        
        # Define initial input
        ground_truth = self.embedding(targets)
        ground_truth_embedded = self.dense(ground_truth)

        # Initialize output container
        decoder_outputs = Variable(torch.zeros(self.time_steps,self.batch_size,self.num_tokens))  # (time_steps, batch_size, vocab_size)

        # Real targets
        b = targets.size(1)
        t = targets.size(0)
        real_targets = Variable(torch.zeros(t,b)).to(self.device) # (time_steps,batch_size)
        real_targets[:-1,:] = targets[1:,:]

        pg_loss = 0.
        
        # Unfold the decoder RNN on the time dimension
        for t in range(self.time_steps):
            outputs1, hidden1 = self.lstm1(torch.zeros(1, self.batch_size, self.latent_dim).to(self.device), hidden1)
            outputs2, hidden2 = self.lstm2(torch.cat([ground_truth_embedded[t],outputs1],-1),hidden2) # (1, batch_size, latent_dim)
            
            outputs2 = outputs2.squeeze(0)  # squeeze the time dimension (batch_size, latent_dim)
            outputs2 = self.out(outputs2)  # (batch_size, vocab_size)
            decoder_outputs[t] = outputs2

            # Calculate the one-step Loss
            loss = self.criterion(outputs2, real_targets[t].long())
            pg_current_loss = loss * rewards[t]
            pg_loss = pg_loss + pg_current_loss
        
        return decoder_outputs, hidden1, hidden2, pg_loss / opt.mxlen
    
    # For decoding state
    def evaluation(self, inputs):
        
        # Encoder state
        _, encoder_hidden1, encoder_hidden2 = self.encoder_forward(inputs)
        hidden1 = encoder_hidden1
        hidden2 = encoder_hidden2
        
        # Define initial input
        decoder_input = Variable(torch.LongTensor([[self.sos_id]*self.batch_size])).to(self.device)
        decoder_input_embedded = self.embedding(decoder_input)
        decoder_input_embedded = self.dense(decoder_input_embedded)
        
        # Initialize output container
        decoder_outputs = Variable(torch.zeros(self.time_steps, self.batch_size, self.num_tokens)).to(self.device)  # (time_steps, batch_size, vocab_size)
        decoded_indices = Variable(torch.zeros(self.time_steps, self.batch_size, dtype=torch.int32).long()).to(self.device) # (time_steps, batch_size)

        # Unfold the decoder RNN on the time dimension
        
        for t in range(self.time_steps):
            outputs1, hidden1 = self.lstm1(torch.zeros(1, self.batch_size, self.latent_dim).to(self.device), hidden1)
            outputs2, hidden2 = self.lstm2(torch.cat([decoder_input_embedded,outputs1],-1),hidden2)
            outputs2 = self.out(outputs2)
            decoder_outputs[t] = outputs2

            # Its own last output
            _, indices = torch.topk(decoder_outputs[t], 2)

            # Change <UNK> into second highest probaility word
            for i in range(self.batch_size):
                decoder_input[0][i] = indices[i][0] if indices[i][0] != 3 and indices[i][0] != 0 else indices[i][1]
            
            decoder_input_embedded = self.embedding(decoder_input)
            decoder_input_embedded = self.dense(decoder_input_embedded)
            
            # Store outputs
            decoded_indices[t] = decoder_input
        
        return decoded_indices, hidden1, hidden2
    

        