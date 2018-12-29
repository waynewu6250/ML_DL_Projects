import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam

from models import Encoder
from models import Decoder
from models import Seq2seq
from data import TrainData
from config import opt


def train():
    
    # dataset
    mydata = TrainData(opt.data_path, opt.toks_path)

    # models
    encoder = Encoder(num_encoder_tokens=mydata.data.num_encoder_tokens,
                      char_dim=opt.char_dim,
                      latent_dim=opt.latent_dim)

    decoder = Decoder(num_decoder_tokens=mydata.data.num_decoder_tokens,
                      char_dim=opt.char_dim,
                      latent_dim=opt.latent_dim,
                      teacher_forcing_ratio=opt.teacher_forcing_ratio,
                      sos_id=mydata.data.output_word2id["<START>"])

    seq2seq = Seq2seq(encoder=encoder,
                      decoder=decoder)
    
    if opt.model_path:
        seq2seq.load_state_dict(torch.load(opt.model_path, map_location="cpu"))
        print("Pretrained model has been loaded.\n")
    
    
    #======================================================================#

    optimizer= Adam(seq2seq.parameters(), lr=opt.learning_rate)
    criterion = nn.NLLLoss(ignore_index=0, size_average=True)

    for epoch in range(opt.epochs):
        print("epoch %d:"%epoch)
        mini_batches = mydata._mini_batches(opt.batch_size)
        
        for ii, (ib, tb) in enumerate(mini_batches):
            
            optimizer.zero_grad()
            decoder_outputs, decoder_hidden = seq2seq(ib, tb)
            
            # Reshape the outputs
            b = decoder_outputs.size(1)
            t = decoder_outputs.size(0)
            targets = tb.contiguous().view(-1)  # S = (B*T)
            decoder_outputs = decoder_outputs.view(b * t, -1)  # S = (B*T) x V
            loss = criterion(decoder_outputs, targets)

            if ii % 10 == 0:
                print("Current Loss:",loss.data.item())
            
            loss.backward()
            optimizer.step()
        
        save_path = "checkpoints/epoch-%s.pth"%epoch

        torch.save(seq2seq.state_dict(), save_path)

if __name__ == "__main__":
    train()



            

    
    


