import random
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim import Adam

from models import Encoder
from models import Decoder
from models import Seq2seq
from models import AttnDecoder
from models import AttnSeq2seq
from data import TrainData
from config import opt

def test():
    
    while True:
        data = input('You say: ')
        if data == "exit":
            break
        
        # dataset
        mydata = TrainData(opt.data_path, opt.toks_path)

        # models
        encoder = Encoder(num_encoder_tokens=mydata.data.num_encoder_tokens,
                        char_dim=opt.char_dim,
                        latent_dim=opt.latent_dim)

        if opt.attn:

            decoder = AttnDecoder(num_decoder_tokens=mydata.data.num_decoder_tokens,
                                  char_dim=opt.char_dim,
                                  latent_dim=opt.latent_dim,
                                  time_steps=opt.mxlen,
                                  teacher_forcing_ratio=opt.teacher_forcing_ratio,
                                  sos_id=mydata.data.output_word2id["<START>"],
                                  dropout_rate=0.1)
            seq2seq = AttnSeq2seq(encoder=encoder,
                                  decoder=decoder)
        else:
            decoder = Decoder(num_decoder_tokens=mydata.data.num_decoder_tokens,
                              char_dim=opt.char_dim,
                              latent_dim=opt.latent_dim,
                              teacher_forcing_ratio=opt.teacher_forcing_ratio,
                              sos_id=mydata.data.output_word2id["<START>"])

            seq2seq = Seq2seq(encoder=encoder,
                              decoder=decoder)
        
        encoder_data = mydata._test_batch(data, opt.mxlen)
        if opt.model_path:
            seq2seq.load_state_dict(torch.load(opt.model_path, map_location="cpu"))
        
        decoded_indices = seq2seq.evaluation(encoder_data)
        decoded_sequence = ""
        for ii, idx in enumerate(decoded_indices):
            if ii == 0:
                continue
            else:
                sampled_tok = mydata.data.output_id2word[idx]
                if sampled_tok == "<EOS>":
                    break
                else:
                    decoded_sequence += ' '+sampled_tok
        
        print("WayneBot: ",decoded_sequence)

if __name__ == "__main__":
    test()