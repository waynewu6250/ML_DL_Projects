import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam, RMSprop

from models import Encoder, Decoder, Seq2seq, AttnDecoder, AttnSeq2seq
from models import NewSeq2seq
from data import TrainData
from config import opt


def train():
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    torch.backends.cudnn.enabled = False
    
    
    # dataset
    mydata =  TrainData(opt.data_path, opt.conversation_path, opt.results_path, opt.prev_sent, True)

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
        seq2seq.load_state_dict(torch.load(opt.model_path, map_location='cpu'))
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
            decoder_probs = nn.LogSoftmax(dim=1)(decoder_outputs)
            _, index = torch.topk(decoder_probs[:,0,:], 1, dim=1)

            
            # Reshape the outputs
            b = decoder_outputs.size(1)
            t = decoder_outputs.size(0)
            targets = Variable(torch.zeros(t,b)).to(device) # (time_steps,batch_size)
            targets[:-1,:] = tb[1:,:]
            print("predicted:",index.squeeze(1).numpy())
            print("targets:",(targets[:,0]).long().numpy())
            print(" ".join([mydata.data.id2word[i] for i in index.squeeze(1).numpy() if i != 0]))
            print(" ".join([mydata.data.id2word[i] for i in (targets[:,0]).long().numpy() if i != 0]))

            targets = targets.contiguous().view(-1)  # (time_steps*batch_size)
            decoder_outputs = decoder_outputs.view(b * t, -1)  # S = (time_steps*batch_size) x V
            loss = criterion(decoder_outputs, targets.long())

            if ii % 1 == 0:
                print("Current Loss:",loss.data.item())
            
            loss.backward()
            optimizer.step()
        
        save_path = "checkpoints/epoch-%s.pth"%epoch

        torch.save(seq2seq.state_dict(), save_path)

if __name__ == "__main__":
    train()



            

    
    


