import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, RMSprop

from models import NewSeq2seq, NewSeq2seqAttention, NewSeq2seqRL
from data import TrainData
from config import opt
import numpy as np


#=============================================================#
#                        USEFUL FUNCTION                      #
#=============================================================#

def make_batch(data, input_data, mxlen):
    token_data = [data.text_prepare(sent)[:mxlen] for sent in input_data]
    batch_data = np.zeros((len(input_data), mxlen), dtype='float32')
    for i, sent in enumerate(token_data):
        for t, word in enumerate(sent):
            if word in data.word2id:
                batch_data[i, t] = data.word2id[word]
            else:
                batch_data[i, t] = 3
    batch_data = Variable(torch.LongTensor(batch_data)).transpose(0, 1).to(opt.device)
    return batch_data

def count_rewards(dull_loss, forward_entropy, backward_entropy, forward_target, backward_target, reward_type='pg'):
        if reward_type == 'normal':
            return torch.ones([opt.batch_size, opt.mxlen]).to(opt.device)
        
        if reward_type == 'pg':
            total_loss = torch.zeros([opt.batch_size, opt.mxlen]).to(opt.device)
        
            for i in range(opt.batch_size):
                
                # 1. dull set loss
                total_loss[i, :] += dull_loss[i]

                #2. forward loss
                forward_len = len(forward_target[i].split())
                backward_len = len(backward_target[i].split())
                if forward_len > 0:
                    total_loss[i, :] += (forward_entropy / forward_len)
                if backward_len > 0:
                    total_loss[i, :] += (backward_entropy / backward_len)
                
            total_loss = 1/(1+np.exp(-total_loss)) * 1.1
            
            return total_loss

#=============================================================#
#                        RL TRAIN PHASE                       #
#=============================================================#

def train_RL(**kwargs):

    # attributes
    for k,v in kwargs.items():
        setattr(opt,k,v)
    
    torch.backends.cudnn.enabled = False

    dull_set = ["I don't know what you're talking about.", 
                "I don't know.", 
                "You don't know.", 
                "You know what I mean.", 
                "I know what you mean.", 
                "You know what I'm saying.", 
                "You don't know anything."]
    ones_reward = torch.ones([opt.batch_size, opt.mxlen]).to(self.device)
    
    # dataset
    mydata = TrainData(opt.data_path, opt.conversation_path, opt.results_path, opt.prev_sent, True)
    # Dullset data
    dull_target_data = make_batch(mydata, dull_set, opt.mxlen)

    # models
    # 1. RL model
    seq2seq_rl =  NewSeq2seqRL(num_tokens=mydata.data.num_tokens,
                               opt=opt,
                               sos_id=mydata.data.word2id["<START>"])
    if opt.model_rl_path:
        seq2seq_rl.load_state_dict(torch.load(opt.model_rl_path, map_location="cpu"))
        print("Pretrained model has been loaded.\n")
    seq2seq_rl = seq2seq_rl.to(opt.device)

    # 2. Normal model
    seq2seq_normal =  NewSeq2seq(num_tokens=mydata.data.num_tokens,
                                 opt=opt,
                                 sos_id=mydata.data.word2id["<START>"])
    if opt.model_path:
        seq2seq_normal.load_state_dict(torch.load(opt.model_path, map_location="cpu"))
        print("Pretrained model has been loaded.\n")
    seq2seq_normal = seq2seq_normal.to(opt.device)


    #=============================================================#

    optimizer= RMSprop(seq2seq_rl.parameters(), lr=opt.learning_rate)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(opt.epochs):
        print("epoch %d:"%epoch)
        mini_batches = mydata._mini_batches(opt.batch_size)
        
        for ii, (ib, tb) in enumerate(mini_batches):
            
            ib = ib.to(opt.device)
            tb = tb.to(opt.device)
            
            optimizer.zero_grad()
            
            # First evaluate an output
            action_words, _, _ = seq2seq_rl.evaluate(ib)

            dull_data = seq2seq_rl(action_words, dull_target_data, ones_reward)
            forward_data = seq2seq_rl(ib, action_words, ones_reward)
            backward_data = seq2seq_normal(action_words, ib[:opt.mxlen])
            backward_targets = ib[:opt.mxlen].contiguous().view(-1)  # (time_steps*batch_size)
            backward_outputs = backward_data[0].view(opt.batch_size * opt.mxlen, -1)  # S = (time_steps*batch_size) x V
            backward_loss = criterion(backward_outputs, backward_targets.long())

            rewards = count_rewards(dull_data[3], forward_data[3], backward_loss, action_words, ib[:opt.mxlen])

            # Add rewards to train the data
            decoder_outputs, decoder_hidden1, decoder_hidden2, pg_loss = seq2seq_rl(ib, tb, rewards)
            
            # Reshape the outputs
            b = decoder_outputs.size(1)
            t = decoder_outputs.size(0)
            targets = Variable(torch.zeros(t,b)).to(device) # (time_steps,batch_size)
            targets[:-1,:] = tb[1:,:]

            targets = targets.contiguous().view(-1)  # (time_steps*batch_size)
            decoder_outputs = decoder_outputs.view(b * t, -1)  # S = (time_steps*batch_size) x V
            loss = criterion(decoder_outputs, targets.long())
            
            if ii % 1 == 0:
                print("Current Loss:",loss.data.item())
            
            loss.backward()
            optimizer.step()

            save_path = "checkpoints/rl-epoch-%s.pth"%epoch
            if (epoch+1) % 10 == 0:
                t.save(seq2seq_rl.state_dict(), save_path)














    




    
