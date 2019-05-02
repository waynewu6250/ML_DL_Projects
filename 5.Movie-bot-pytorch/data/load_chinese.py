import torch as t
from torch.autograd import Variable
import numpy as np
import re
import pickle
import h5py
import os

class ChineseData:
    def __init__(self, data_path, results_path, load=True):
        
        self.data_path = data_path
        self.results_path = results_path
        self.mxlen = 20
        self.BAD_SYMBOLS_RE = re.compile(r'[^0-9a-z #+_]')
        self.word_count_threshold = 5

        # dictionary
        """
            count:
            { tok1: count1, tok2: count2, ...}
            word2id:
            { tok1: id1, tok2: id2, ...}
            id2word:
            { id1: tok1, id2: tok2, ...}
        """
        self.word2id = {} # Count all word library
        self.id2word = {} # Reverse word2id
        self.count = {} # Word counts

        # load data
        if load:
            data = self.load_results(results_path)
            self.all_convs = data["all_convs"]
            self.num_data = data["num_data"]
            self.word2id = data["word2id"]
            self.id2word = data["id2word"]
            self.count = data["count"]
            self.encoder_input_data = data["encoder_input_data"]
            self.decoder_input_data = data["decoder_input_data"]
            
        else:
            self.all_convs, self.num_data = self.load_conversations()
            self.encoder_input_data, self.decoder_input_data = self.seq2idx()
            self.save_results(results_path)
        
        self.num_tokens = len(self.word2id)
    
    
    #==================================================#
    #                   Count Words                    #
    #==================================================#
    
    def count_words(self):

        # Special Tokens
        self.word2id["<PAD>"] = 0
        self.word2id["<START>"] = 1
        self.word2id["<EOS>"] = 2
        self.word2id["<UNK>"] = 3
        index = 4

        file = open(self.data_path)
        
        for line in file:
            for word in line.rstrip()[2:].split('/'):
                # Count the words
                if not word in self.count:
                    self.count[word] = 1
                else:
                    self.count[word] += 1
        
        vocab = [w for w,count in self.count.items() if count >= self.word_count_threshold]
        for idx, w in enumerate(vocab):
            self.word2id[w] = idx+index
        
        # Arrange word2id and id2word
        self.word2id = {key: i for i, key in enumerate(self.word2id.keys())}
        self.id2word = {i:symbol for symbol, i in self.word2id.items()}
    
    #==================================================#
    #                    Load Text                     #
    #==================================================#
    
    def load_conversations(self):
        """
            Load the chinese file "dgk_shooter_min.conv".
            The ﬁle has two seperate symbols: E represents next data and M represents data:
            E
            M 呵/呵
            M 是/王/若/猫/的/。
            E
            ...
        """
        def func(tok):
            if self.count[tok] <= self.word_count_threshold:
                return "<UNK>"
            else: return tok
        
        self.count_words()
        
        file = open(self.data_path, 'r')
        all_convs = []
        num_data = 0
        
        con_a = []

        for line in file:
            
            if line.rstrip()[0] == 'E':
                con_a = []
                continue
            
            toks = line.rstrip()[2:].split('/')
            con_b = ["<START>"]+list(map(func,toks))[:self.mxlen-2]+["<EOS>"]
            
            # No prev sentence, pass
            if con_a == []:
                con_a = con_b
            else:
                all_convs.append((con_a, con_b))
                num_data += 1
                con_a = con_b
        
        return all_convs, num_data
    
    #==================================================#
    #                    Make Data                     #
    #==================================================#
    
    def seq2idx(self):

        encoder_input_data = np.zeros((self.num_data, self.mxlen), dtype='float32')
        decoder_input_data = np.zeros((self.num_data, self.mxlen), dtype='float32')

        # Make encoder_input_data, decoder_input_data, decoder_target_data
        for i, (input_text, output_text) in enumerate(self.all_convs):
            for t, word in enumerate(input_text):
                encoder_input_data[i, t] = self.word2id[word]
            for t, word in enumerate(output_text):
                decoder_input_data[i, t] = self.word2id[word]
        
        return encoder_input_data, decoder_input_data

    #==================================================#
    #                    Save Files                    #
    #==================================================#

    def save_results(self, file_name):
        data = {"all_convs": self.all_convs,
                "num_data": self.num_data,
                "word2id": self.word2id,
                "id2word": self.id2word,
                "count": self.count,
                "encoder_input_data": self.encoder_input_data,
                "decoder_input_data": self.decoder_input_data}
        with open(file_name, 'wb') as f:
            pickle.dump(data, f)
    
    def load_results(self, file_name):
        with open(file_name, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    
    data = ChineseData("new_data.conv", "chinese_data_new.bin", False)
    
    print(data.encoder_input_data[11])
    print(data.decoder_input_data[11])
    print(data.all_convs[11])
    print(len(data.word2id))
    print(data.num_data)