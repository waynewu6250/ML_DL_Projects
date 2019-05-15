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

def count_rewards(dull_loss, forward_entropy, backward_entropy, reward_type='pg'):
    if reward_type == 'normal':
        return torch.ones(opt.mxlen).to(opt.device)
    
    if reward_type == 'pg':
        total_loss = torch.zeros(opt.mxlen).to(opt.device)
        
        total_loss[:] += (dull_loss+forward_entropy+backward_entropy)/3
        print(total_loss)
            
        total_loss = torch.sigmoid(total_loss) * 1.1
        
        return total_loss

#=============================================================#
#                        RL TRAIN PHASE                       #
#=============================================================#

def train_RL():
    
    torch.backends.cudnn.enabled = False

    dull_set = ["I don't know what you're talking about.", 
                "I don't know.", 
                "You don't know.", 
                "You know what I mean.", 
                "I know what you mean.", 
                "You know what I'm saying.", 
                "You don't know anything."]
    ones_reward = torch.ones(opt.mxlen).to(opt.device)
    
    # dataset
    mydata = TrainData(opt.data_path, opt.conversation_path, opt.results_path, chinese=False, prev_sent=opt.prev_sent)
    
    # Ease of answering data
    dull_target_set = make_batch(mydata.data, dull_set, opt.mxlen)
    dull_target_set = dull_target_set.permute(1,0)
    dull_target_data = Variable(torch.LongTensor(opt.batch_size, opt.mxlen)).to(opt.device)
    for i in range(opt.batch_size):
        dull_target_data[i] = dull_target_set[np.random.randint(len(dull_set))]
    dull_target_data = dull_target_data.permute(1,0)

    # models
    # 1. RL model
    seq2seq_rl =  NewSeq2seqRL(num_tokens=mydata.data.num_tokens,
                               opt=opt,
                               sos_id=mydata.data.word2id["<START>"])
    if opt.model_rl_path:
        seq2seq_rl.load_state_dict(torch.load(opt.model_rl_path, map_location="cpu"))
        print("Pretrained RL model has been loaded.")
    seq2seq_rl = seq2seq_rl.to(opt.device)

    # 2. Normal model
    seq2seq_normal =  NewSeq2seq(num_tokens=mydata.data.num_tokens,
                                 opt=opt,
                                 sos_id=mydata.data.word2id["<START>"])
    if opt.model_path:
        seq2seq_normal.load_state_dict(torch.load(opt.model_path, map_location="cpu"))
        print("Pretrained Normal model has been loaded.")
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
            action_words, _, _ = seq2seq_rl.evaluation(ib)

            # Ease of answering data
            dull_data = seq2seq_rl(action_words, dull_target_data, ones_reward)
            
            # Semantic Coherence: Forward
            forward_data = seq2seq_rl(ib, action_words, ones_reward)

            # Semantic Coherence: Backward
            backward_data = seq2seq_normal(action_words, ib[:opt.mxlen])
            backward_targets = ib[:opt.mxlen].contiguous().view(-1)  # (time_steps*batch_size)
            backward_outputs = backward_data[0].view(opt.batch_size * opt.mxlen, -1)  # S = (time_steps*batch_size) x V
            backward_loss = criterion(backward_outputs, backward_targets.long())

            rewards = count_rewards(dull_data[3], forward_data[3], backward_loss)

            # Add rewards to train the data
            decoder_outputs, decoder_hidden1, decoder_hidden2, pg_loss = seq2seq_rl(ib, tb, rewards)
            
            if ii % 1 == 0:
                print("Current Loss:",pg_loss.data.item())
            
            pg_loss.backward()
            optimizer.step()

            save_path = "checkpoints/rl-epoch-%s.pth"%epoch
            if (epoch+1) % 10 == 0:
                t.save(seq2seq_rl.state_dict(), save_path)

if __name__ == "__main__":
    train_RL()














    




    
