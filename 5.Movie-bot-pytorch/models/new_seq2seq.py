import torch
import torch.nn as nn
from torch.autograd import Variable
from config import opt

class NewSeq2seq(nn.Module):
    def __init__(self, num_tokens, opt, sos_id):

        super(NewSeq2seq, self).__init__()
        self.embedding = nn.Embedding(num_tokens, opt.char_dim)
        self.dense = nn.Linear(opt.char_dim, opt.latent_dim)
        self.lstm1 = nn.LSTM(opt.latent_dim, opt.latent_dim)
        self.lstm2 = nn.LSTM(2*opt.latent_dim, opt.latent_dim)
        
        self.out = nn.Linear(opt.latent_dim, num_tokens)
        self.log_softmax = nn.LogSoftmax(dim=1)  # work with NLLLoss = CrossEntropyLoss
        
        self.char_dim = opt.char_dim
        self.latent_dim = opt.latent_dim
        self.num_tokens = num_tokens
        self.time_steps = opt.mxlen
        self.sos_id = sos_id
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    
    
    def encoder_forward(self, input_var, hidden1=None, hidden2=None):
        """
        input_var: (time_steps, batch_size)
        embedded: (time_steps, batch_size, char_dim)
        outputs, hidden: (time_steps, batch_size, latent_dim)
        """
        self.batch_size = input_var.shape[1]
        embedded = self.embedding(input_var)
        embedded = self.dense(embedded)
        outputs1, hidden1 = self.lstm1(embedded, hidden1)
        padding = torch.zeros(2*opt.mxlen, self.batch_size, self.latent_dim).to(self.device)
        outputs2, hidden2 = self.lstm2(torch.cat([padding,outputs1],-1),hidden2)
        return outputs2, hidden1, hidden2
    
    
    def decoder_forward(self, encoder_hidden1, encoder_hidden2, targets):
        """
        encoder_hidden: (1, batch_size, latent_dim)
        targets: (time_steps, batch_size, 1)
        outputs, hidden: (time_steps, batch_size, latent_dim)
        """
        ground_truth = self.embedding(targets)
        ground_truth_embedded = self.dense(ground_truth)
        padding = torch.zeros(opt.mxlen, self.batch_size, self.latent_dim).to(self.device)
        outputs1, hidden1 = self.lstm1(padding, encoder_hidden1)
        outputs2, hidden2 = self.lstm2(torch.cat([ground_truth_embedded,outputs1],-1), encoder_hidden2)
        outputs2 = self.out(outputs2)

        return outputs2, hidden1, hidden2
    
    
    def forward(self, inputs, targets):
        _, encoder_hidden1, encoder_hidden2 = self.encoder_forward(inputs)
        decoder_outputs, decoder_hidden1, decoder_hidden2 = self.decoder_forward(encoder_hidden1, encoder_hidden2, targets)
        return decoder_outputs, decoder_hidden1, decoder_hidden2
    
    
    # For decoding state
    def evaluation(self, inputs):
        
        # Encoder state
        _, encoder_hidden1, encoder_hidden2 = self.encoder_forward(inputs)
        hidden1 = encoder_hidden1
        hidden2 = encoder_hidden2
        
        # Define initial input
        decoder_input = Variable(torch.LongTensor([[self.sos_id]])).to(self.device)
        decoder_input_embedded = self.embedding(decoder_input)
        decoder_input_embedded = self.dense(decoder_input_embedded)
        
        # Initialize output container
        decoder_outputs = Variable(torch.zeros(self.time_steps,1,self.num_tokens))  # (time_steps, batch_size, vocab_size)
        decoded_indices = []

        # Unfold the decoder RNN on the time dimension
        
        for t in range(self.time_steps):
            outputs1, hidden1 = self.lstm1(torch.zeros(1, 1, self.latent_dim).to(self.device), hidden1)
            outputs2, hidden2 = self.lstm2(torch.cat([decoder_input_embedded,outputs1],-1),hidden2)
            outputs2 = self.out(outputs2)
            decoder_outputs[t] = outputs2

            # Its own last output
            _, indices = torch.topk(decoder_outputs[t], 2)

            # Change <UNK> into second highest probaility word
            index = indices[0][0] if indices[0][0] != 3 and indices[0][0] != 0 else indices[0][1]
            
            decoder_input = index.view(1,-1)
            decoder_input_embedded = self.embedding(decoder_input)
            decoder_input_embedded = self.dense(decoder_input_embedded)
            
            # Store outputs
            decoded_indices.append(index.item())
        
        return decoded_indices, hidden1, hidden2