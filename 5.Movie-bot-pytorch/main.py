import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, RMSprop

from models import Encoder, Decoder, Seq2seq, AttnDecoder, AttnSeq2seq
from models import NewSeq2seq
from data import TrainData
from config import opt

from opencc import OpenCC

def convert(text, mode):
    cc = OpenCC(mode)
    return cc.convert(text)

def train(**kwargs):

    # attributes
    for k,v in kwargs.items():
        setattr(opt,k,v)
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.enabled = False
    
    # dataset
    if opt.chinese:
        mydata = TrainData(opt.chinese_data_path, opt.conversation_path, opt.chinese_results_path, opt.prev_sent, True)
    else:
        mydata = TrainData(opt.data_path, opt.conversation_path, opt.results_path, opt.prev_sent, True)

    # models
    seq2seq =  NewSeq2seq(num_tokens=mydata.data.num_tokens,
                          opt=opt,
                          sos_id=mydata.data.word2id["<START>"])
    # encoder = Encoder(num_encoder_tokens=mydata.data.num_tokens,
    #                   char_dim=opt.char_dim,
    #                   latent_dim=opt.latent_dim)

    # if opt.attn:
    #     decoder = AttnDecoder(num_decoder_tokens=mydata.data.num_tokens,
    #                           char_dim=opt.char_dim,
    #                           latent_dim=opt.latent_dim,
    #                           time_steps=opt.mxlen,
    #                           teacher_forcing_ratio=opt.teacher_forcing_ratio,
    #                           sos_id=mydata.data.word2id["<START>"],
    #                           dropout_rate=0.1)
        
    #     seq2seq = AttnSeq2seq(encoder=encoder,
    #                           decoder=decoder)
    # else:
    #     decoder = Decoder(num_decoder_tokens=mydata.data.num_tokens,
    #                       char_dim=opt.char_dim,
    #                       latent_dim=opt.latent_dim,
    #                       teacher_forcing_ratio=opt.teacher_forcing_ratio,
    #                       sos_id=mydata.data.word2id["<START>"])
                   
    #     seq2seq = Seq2seq(encoder=encoder,
    #                       decoder=decoder)
    
    if opt.model_path:
        if opt.chinese:
            seq2seq.load_state_dict(torch.load(opt.chinese_model_path, map_location="cpu"))
        else:
            seq2seq.load_state_dict(torch.load(opt.model_path, map_location="cpu"))
        print("Pretrained model has been loaded.\n")
    
    seq2seq = seq2seq.to(device)
    
    
    #======================================================================#

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
#=============================================================#
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
            mydata = TrainData(opt.chinese_data_path, opt.conversation_path, opt.chinese_results_path, opt.chinese, opt.prev_sent, True)
        else:
            data = ' '.join(data.split(' '))
            mydata = TrainData(opt.data_path, opt.conversation_path, opt.results_path, opt.chinese, opt.prev_sent, True)

        # models
        seq2seq =  NewSeq2seq(num_tokens=mydata.data.num_tokens,
                              opt=opt,
                              sos_id=mydata.data.word2id["<START>"])
        
        # encoder = Encoder(num_encoder_tokens=mydata.data.num_tokens,
        #                   char_dim=opt.char_dim,
        #                   latent_dim=opt.latent_dim)

        # if opt.attn:

        #     decoder = AttnDecoder(num_decoder_tokens=mydata.data.num_tokens,
        #                           char_dim=opt.char_dim,
        #                           latent_dim=opt.latent_dim,
        #                           time_steps=opt.mxlen,
        #                           teacher_forcing_ratio=opt.teacher_forcing_ratio,
        #                           sos_id=mydata.data.word2id["<START>"],
        #                           dropout_rate=0.1)
        #     seq2seq = AttnSeq2seq(encoder=encoder,
        #                           decoder=decoder)
        # else:
        #     decoder = Decoder(num_decoder_tokens=mydata.data.num_tokens,
        #                       char_dim=opt.char_dim,
        #                       latent_dim=opt.latent_dim,
        #                       teacher_forcing_ratio=opt.teacher_forcing_ratio,
        #                       sos_id=mydata.data.word2id["<START>"])

        #     seq2seq = Seq2seq(encoder=encoder,
        #                       decoder=decoder)
        
        if opt.model_path:
            if opt.chinese:
                seq2seq.load_state_dict(torch.load(opt.chinese_model_path, map_location="cpu"))
            else:
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
            else convert(decoded_sequence,'s2t').replace("雞仔","我").replace("主人","跟你說").replace("主子哦","").replace("主子","跟你說"))
        prev_sentence = decoded_sequence

if __name__ == "__main__":
    import fire
    fire.Fire()



            

    
    


