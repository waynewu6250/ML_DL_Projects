import torch as t
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence
from utils import CaptionGenerator

class CaptionModel(nn.Module):
    def __init__(self, opt, word2ix, ix2word):
        super(CaptionModel,self).__init__()
        self.opt = opt
        self.word2ix = word2ix
        self.ix2word = ix2word

        self.fc = nn.Linear(2048, opt.hidden_dim) #for imgs embedding
        self.embedding = nn.Embedding(len(word2ix), opt.embedding_dim)
        self.rnn = nn.LSTM(opt.embedding_dim, opt.hidden_dim, num_layers=opt.num_layers)
        self.fc2 = nn.Linear(opt.hidden_dim, len(word2ix))

    def forward(self, img_feats, captions, lengths):
        
        # Concatenate img and caption together
        img_feats = self.fc(img_feats).unsqueeze(0) #size turns to (1, batch_size, embedding_dim)
        embeddings = self.embedding(captions) #(time_steps, batch_size, embedding_dim)
        embeddings = t.cat([img_feats, embeddings],0)
        pack_embeddings = pack_padded_sequence(embeddings, lengths)
        
        # Feed into rnn
        outputs, states = self.rnn(pack_embeddings)
        pred = self.fc2(outputs[0])
        return pred, states
    
    def generate(self, img,
                 beam_size=3,
                 max_caption_length=20,
                 length_normalization_factor=0.0):
        """
        Generate images by tensorflow beam search algorithms
        """
        cap_gen = CaptionGenerator(embedder=self.embedding,
                                   rnn=self.rnn,
                                   classifier=self.fc2,
                                   eos_id=self.word2ix['<EOS>'],
                                   beam_size=beam_size,
                                   max_caption_length=max_caption_length,
                                   length_normalization_factor=length_normalization_factor)
        if next(self.parameters()).is_cuda:
            img = img.cuda()
        img =img.unsqueeze(0)
        img = self.fc(img).unsqueeze(0)
        sentences, score = cap_gen.beam_search(img)
        sentences = [' '.join([self.ix2word[idx.item()] for idx in sent])
                     for sent in sentences]
        return sentences
    

        