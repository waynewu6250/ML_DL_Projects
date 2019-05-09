import torch as t
from torch.autograd import Variable
import numpy as np
import re
import pickle
import h5py
import os

class Data:
    def __init__(self, data_path, conversation_path, results_path, prev_sent=2, load=True):
        
        self.data_path = data_path
        self.conversation_path = conversation_path
        self.results_path = results_path
        self.prev_sent = prev_sent
        self.mxlen = 20
        self.REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
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
            self.conv_ids = data["conv_ids"]
            self.num_data = data["num_data"]
            self.num_conv = data["num_conv"]
            self.word2id = data["word2id"]
            self.id2word = data["id2word"]
            self.count = data["count"]
            self.encoder_input_data = data["encoder_input_data"]
            self.decoder_input_data = data["decoder_input_data"]
            
        else:
            self.all_convs, self.conv_ids, self.num_data, self.num_conv = self.load_conversations(self.load_text(), prev_sent)
            self.encoder_input_data, self.decoder_input_data = self.seq2idx()
            self.save_results(results_path)
        
        self.num_tokens = len(self.word2id)
        
    
    
    #==================================================#
    #                   Text Prepare                   #
    #==================================================#

    def text_prepare(self, text):
        """
            text: a string
            
            return: modified string tokens 
                    [tok1, tok2 , ...] which is a single sentence from one character
        """
        tok = ["<START>"] # add START token to represent sentence start
        text = text.lower() # lowercase text
        text = re.sub(self.REPLACE_BY_SPACE_RE, ' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = re.sub(self.BAD_SYMBOLS_RE, '', text) # delete symbols which are in BAD_SYMBOLS_RE from text
        tok += (text.split()+["<EOS>"]) # add EOS token to represent sentence end
        
        return tok
    
    #==================================================#
    #                   Count Words                    #
    #==================================================#
    
    def count_words(self, line_dict):
        
        # Special Tokens
        self.word2id["<PAD>"] = 0
        self.word2id["<START>"] = 1
        self.word2id["<EOS>"] = 2
        self.word2id["<UNK>"] = 3
        index = 4
        
        for toks in line_dict.values():
            for word in toks:
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
    
    def load_text(self):
        """
            Load the movie_lines.tsv file which contains the data. 
            The ﬁle has ﬁve tab separated columns containing the following ﬁelds:
            1. lineID
            2. characterID (who uttered this phrase)
            3. movieID
            4. character name
            5. text of the utterance
            
            Here we only extract lineID and utterance
            
            line_dict = {lineID1: utterance1,
                        lineID2: utterance2,
                        lineID3: utterance3,
                        ...}
        """
        file = open(self.data_path)
        line_dict = {}
        
        for line in file:
            cols = line.rstrip().split("\t")
            line_dict[cols[0].replace('"','')] = self.text_prepare(cols[-1])
        
        self.count_words(line_dict)
        
        return line_dict
    
    #==================================================#
    #                Load Conversations                #
    #==================================================#
    
    def load_conversations(self, line_dict, prev_sent=2):
        """
            Load movie_conversations.txt which has the conversation lists
            all_convs = [converation 1: [('', sent1, sent2),
                                        (sent1, sent2, sent3),
                                        (sent2, sent3, sent4),
                                        ...]]
            num_data: number of all data
            num_conv: number of all conversation pairs
        
        """
        def func(tok):
            if self.count[tok] <= self.word_count_threshold:
                return "<UNK>"
            else: return tok
        
        file = open(self.conversation_path, 'r').read().split('\n')[:-1]
        all_convs = []
        conv_ids = []
        num_data, num_conv = 0, 0

        for i,conv in enumerate(file):
            DELETE = re.compile('[/(){}\[\]\|@,;\']')
            conv = re.sub(DELETE, '', conv.split(' +++$+++ ')[-1]).split()
            
            con_a_1 = []
            conv_ids.append(num_data)
            for i in range(len(conv)-1):
                con_a_2 = list(map(func,line_dict[conv[i]]))[:self.mxlen]
                con_b = list(map(func,line_dict[conv[i+1]]))[:self.mxlen]
                all_convs.append((con_a_1+con_a_2, con_b) if prev_sent==2 else (con_a_2, con_b))
                num_data += 1
                con_a_1 = con_a_2
                
            num_conv += 1
        
        return all_convs, conv_ids, num_data, num_conv
    
    #==================================================#
    #                    Make Data                     #
    #==================================================#
    
    def seq2idx(self):

        encoder_input_data = np.zeros((self.num_data, self.mxlen*self.prev_sent), dtype='float32')
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
                "conv_ids": self.conv_ids,
                "num_data": self.num_data,
                "num_conv": self.num_conv,
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
    
    data = Data("movie_lines.tsv", "movie_conversations.txt", "data.bin", 2, False)
    print(data.encoder_input_data[11])
    print(data.decoder_input_data[11])
    print(data.all_convs[11])