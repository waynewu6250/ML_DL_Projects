import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, RMSprop

from models import NewSeq2seq, NewSeq2seqAttention
from data import TrainData
from config import opt

from opencc import OpenCC

def convert(text, mode):
    cc = OpenCC(mode)
    return cc.convert(text)

#=============================================================#
#                     NORMAL TRAIN PHASE                      #
#=============================================================#

def train(**kwargs):

    # attributes
    for k,v in kwargs.items():
        setattr(opt,k,v)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.enabled = False
    
    # dataset
    if opt.chinese:
        mydata = TrainData(opt.chinese_data_path, opt.conversation_path, opt.chinese_results_path, opt.chinese, opt.fb, opt.prev_sent, True)
    else:
        mydata = TrainData(opt.data_path, opt.conversation_path, opt.results_path, opt.chinese, opt.fb, opt.prev_sent, True)

    # models
    if opt.attn:
        seq2seq =  NewSeq2seqAttention(num_tokens=mydata.data.num_tokens,
                                       opt=opt,
                                       sos_id=mydata.data.word2id["<START>"])
        if opt.model_path:
            
            pretrained_dict = torch.load(opt.model_path, map_location="cpu")
            model_dict = seq2seq.state_dict()
            
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            model_dict.update(pretrained_dict)
            
            seq2seq.load_state_dict(model_dict)
            print("Transfer pretrained model to current model.\n")
        
        if opt.model_attention_path:
            seq2seq.load_state_dict(torch.load(opt.model_attention_path, map_location="cpu"))
            print("Pretrained model has been loaded.\n")
    else:
        seq2seq =  NewSeq2seq(num_tokens=mydata.data.num_tokens,
                              opt=opt,
                              sos_id=mydata.data.word2id["<START>"])
        
        if opt.chinese:
            if opt.chinese_model_path:
                seq2seq.load_state_dict(torch.load(opt.chinese_model_path, map_location="cpu"))
                print("Pretrained model has been loaded.\n")
        else:
            if opt.model_path:
                seq2seq.load_state_dict(torch.load(opt.model_path, map_location="cpu"))
                print("Pretrained model has been loaded.\n")
    
    seq2seq = seq2seq.to(device)
    
    
    #=============================================================#

    optimizer= RMSprop(seq2seq.parameters(), lr=opt.learning_rate)
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(opt.epochs):
        print("epoch %d:"%epoch)
        mini_batches = mydata._mini_batches(opt.batch_size)
        
        for ii, (ib, tb) in enumerate(mini_batches):
            
            ib = ib.to(device)
            tb = tb.to(device)
            
            optimizer.zero_grad()
            decoder_outputs, decoder_hidden1, decoder_hidden2 = seq2seq(ib, tb)
            
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
        if opt.chinese:
            save_path = "checkpoints/chinese-epoch-%s.pth"%epoch
        else:
            save_path = "checkpoints/epoch-%s.pth"%epoch

        torch.save(seq2seq.state_dict(), save_path)



#=============================================================#
#                         TEST PHASE                          #
#=============================================================#


def test(**kwargs):

    # attributes
    for k,v in kwargs.items():
        setattr(opt,k,v)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.enabled = False

    prev_sentence = ""

    while True:
        
        data = input('You say: ')
        if data == "exit":
            break
        if opt.prev_sent == 2:
            data = (prev_sentence + data) if not opt.chinese else data
        
        # Dataset
        if opt.chinese:
            data = list(convert(data, 't2s'))
            mydata = TrainData(opt.chinese_data_path, opt.conversation_path, opt.chinese_results_path, opt.chinese, opt.fb, opt.prev_sent, True)
        else:
            data = ' '.join(data.split(' '))
            mydata = TrainData(opt.data_path, opt.conversation_path, opt.results_path, opt.chinese, opt.fb, opt.prev_sent, True)

        # models
        if opt.attn:
            seq2seq =  NewSeq2seqAttention(num_tokens=mydata.data.num_tokens,
                                        opt=opt,
                                        sos_id=mydata.data.word2id["<START>"])
            if opt.model_attention_path:
                seq2seq.load_state_dict(torch.load(opt.model_attention_path, map_location="cpu"))
        else:
            seq2seq =  NewSeq2seq(num_tokens=mydata.data.num_tokens,
                              opt=opt,
                              sos_id=mydata.data.word2id["<START>"])
        
            if opt.chinese:
                if opt.chinese_model_path:
                    seq2seq.load_state_dict(torch.load(opt.chinese_model_path, map_location="cpu"))
            else:
                if opt.model_path:
                    seq2seq.load_state_dict(torch.load(opt.model_path, map_location="cpu"))
        
        seq2seq = seq2seq.to(device)

        # Predict
        encoder_data = mydata._test_batch(data, 2*opt.mxlen if not opt.chinese else opt.mxlen).to(device)
        decoded_indices, decoder_hidden1, decoder_hidden2 = seq2seq.evaluation(encoder_data)
        
        toks_to_replace = {"i":"I","im":"I'm","id":"I'd","ill":"I'll","iv":"I'v","hes":"he's","shes":"she's",
                           "youre":"you're","its":"it's","dont":"don't","youd":"you'd","cant":"can't","thats":"that's",
                           "isnt":"isn't","didnt":"didn't","hows":"how's","ive":"I've"}

        decoded_sequence = ""
        for idx in decoded_indices:
            sampled_tok = mydata.data.id2word[idx]
            if sampled_tok == "<START>":
                continue
            elif sampled_tok == "<EOS>":
                break
            else:
                if not opt.chinese:
                    if sampled_tok in toks_to_replace:
                        sampled_tok = toks_to_replace[sampled_tok]
                    decoded_sequence += sampled_tok+' '
                else:
                    decoded_sequence += sampled_tok
        
        print("WayneBot:",decoded_sequence if not opt.chinese \
            else convert(decoded_sequence,'s2t').replace("雞仔","我").replace("主人","陛下").replace("主子","陛下"))
        prev_sentence = decoded_sequence

if __name__ == "__main__":
    import fire
    fire.Fire()



            

    
    


