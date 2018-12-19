import torch as t
from torch.autograd import Variable
import numpy as np
import re
import pickle
import h5py
from data.load import Data, Vocabulary

class TrainData:
    
    def __init__(self, data_path, toks_path):
        
        self.vocab = Vocabulary(data_path)
        all_lineids, all_ids, _ = self.vocab.load_data()
        
        # Load the processed data to save time
        file1=open(toks_path,"rb")
        all_toks_new=pickle.load(file1)
        
        self.data = Data(all_lineids, all_ids, all_toks_new)
    
    def _mini_batches(self, batch_size):
        
        self.indices_sequences = [(i,j) for i,j in zip(self.data.encoder_input_data, self.data.decoder_input_data)]
        np.random.shuffle(self.indices_sequences)
        mini_batches = [self.indices_sequences[k: k + batch_size] for k in range(0, len(self.indices_sequences), batch_size)]

        for batch in mini_batches:
            seq_pairs = sorted(batch, key=lambda seqs: len(seqs[0]), reverse=True)  # sorted by input_lengths
            input_seqs = [pair[0] for pair in seq_pairs]
            target_seqs = [pair[1] for pair in seq_pairs]
            
            input_var = Variable(t.LongTensor(input_seqs)).transpose(0, 1)  # time * batch
            target_var = Variable(t.LongTensor(target_seqs)).transpose(0, 1)  # time * batch

            yield (input_var, target_var)
    
    # For evaluation state
    def tokenize_seq(self, input_data, mxlen):
        
        token_data = self.vocab.text_prepare(input_data)
        encoder_data = np.zeros((1, mxlen), dtype='float32')

        for t, word in enumerate(token_data):
            if word in self.data.input_word2id:
                encoder_data[0, t] = self.data.input_word2id[word]
            else:
                encoder_data[0, t] = 2
        return encoder_data
    
    def _test_batch(self, input_data, mxlen):

        encoder_data = self.tokenize_seq(input_data, mxlen)
        input_var = Variable(t.LongTensor(encoder_data)).transpose(0, 1)

        return input_var



# a = TrainData("movie_lines.tsv", "all_toks_new.bin")
# for ib, tb in a._mini_batches(10):
#     print(ib, tb)
#     break