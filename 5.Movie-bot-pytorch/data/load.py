import torch as t
from torch.autograd import Variable
import numpy as np
import re
import pickle
import h5py

class Vocabulary:
    def __init__(self, data_path):
        
        self.data_path = data_path
        self.mxlen = 20

        # dictionary
        self.word2id = {} # Count all word library
        self.id2word = {} # Reverse word2id
        
        self.REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
        self.BAD_SYMBOLS_RE = re.compile(r'[^0-9a-z #+_]')
    
    def load_data(self):
        """
            Load the movie_lines.tsv file which contains the data. 
            The ﬁle has ﬁve tab separated columns containing the following ﬁelds:
            1. lineID
            2. characterID (who uttered this phrase)
            3. movieID
            4. character name
            5. text of the utterance
            
            all_lineids = [lineids1, lineids2, ...] where lineids is a sequence of 
                utterances for one movie.
            all_ids = [ids1, ids2, ...] where ids is a sequence of 
                character ids for one movie.
            all_toks = [toks1, toks2, ...] where toks is a sequence of 
                words (sentences) for one movie.
        
        """
        file = open(self.data_path)
        all_lineids = []
        all_ids = []
        all_toks = []
        
        lineids = []
        ids = []
        toks = []
        mid = "m0"
        i = 0
        
        for line in file:
            cols = line.rstrip().split("\t")
            #Only get the data with entire five columns
            if len(cols) < 5:
                continue
            
            if cols[2] != mid:
                all_lineids.append(lineids)
                all_ids.append(ids)
                all_toks.append(toks)
                
                #Restart new movie data
                lineids = [int(cols[0].strip('"L'))]
                ids = [int(cols[1].strip('u'))]
                toks = [cols[4]]
                mid = cols[2]
                continue
            
            lineids.append(int(cols[0].strip('"L')))
            ids.append(int(cols[1].strip('u')))
            toks.append(cols[4])
            
            i += 1
            if i == 103085:
                break
        
        if len(toks) > 0:
            all_lineids.append(lineids)
            all_ids.append(ids)
            all_toks.append(toks)
        
        return all_lineids, all_ids, all_toks
    
    #==================================================#
    #                   Text Prepare                   #
    #==================================================#

    # 1) Remove bad symbols and tokenization
    def text_prepare(self, txt):
        """
            text: a string
            
            return: modified string tokens 
                    [tok1, tok2 , ...] which is a single sentence from one character
        """
        tok = ["<START>"] # add START token to represent sentence start
        txt = txt.lower() # lowercase text
        txt = re.sub(self.REPLACE_BY_SPACE_RE, ' ', txt) # replace REPLACE_BY_SPACE_RE symbols by space in text
        txt = re.sub(self.BAD_SYMBOLS_RE, '', txt) # delete symbols which are in BAD_SYMBOLS_RE from text
        tok += (txt.split()+["<EOS>"]) # add EOS token to represent sentence end
        
        return tok
    
    # 2) Dictionary of all words from train corpus with their counts.
    #    Dictionary of all words with its ids
    def count_words(self, all_toks):
        """
        count:
        { tok1: count1, tok2: count2, ...}
        word2id:
        { tok1: id1, tok2: id2, ...}
        id2word:
        { id1: tok1, id2: tok2, ...}
        
        """
        count = {}
        
        # Special Tokens
        self.word2id["<START>"] = 0
        self.word2id["<EOS>"] = 1
        self.word2id["<UNK>"] = 2
        index = 3
        
        for toks in all_toks:
            for tok in toks:
                for word in tok:
                    # Count the words
                    if not word in count:
                        count[word] = 1
                    else:
                        count[word] += 1
                    # Make dictionary
                    if not word in self.word2id:
                        self.word2id[word] = index
                        index += 1
        
        return count

    def text_tokenize(self, all_toks):
    
        """
        all_toks_new: 
        [
            movie 0:[ line 0: [tok1, tok2, ...],
                    line 1: [tok1, tok2, ...],
                    ... ]
            movie 1:[ line 0: [tok1, tok2, ...],
                    line 1: [tok1, tok2, ...],
                    ... ]
            movie 2:[ line 0: [tok1, tok2, ...],
                    line 1: [tok1, tok2, ...],
                    ... ]
            ...

        ]
        
        scarce_words_counts: a list of words that appear only once.
        [tok1, tok2, tok3, ...]
        
        """
        all_toks_new = []

        # Prepare the text
        for toks in all_toks:
            toks = [self.text_prepare(x) for x in toks]
            all_toks_new.append(toks)

        # Count the words that appears only once.
        words_counts = self.count_words(all_toks_new)
        scarce_words_counts = [x[0] for x in sorted(words_counts.items(), key = lambda x: x[1], reverse=True) if x[1] == 1]
        
        # Remove scarce words in word2id dictionary and reindex all words
        for word in scarce_words_counts:
            del self.word2id[word]
        
        # Arrange word2id and id2word
        self.word2id = {key: i for i, key in enumerate(self.word2id.keys())}
        self.id2word = {i:symbol for symbol, i in self.word2id.items()}
        
        return all_toks_new, scarce_words_counts
    
    #3. replace scarce tokens and restrict word length
    def modify(self, all_toks_new, scarce_words_counts):
        """
        all_toks_new: (each with same length mxlen)
        [
            movie 0:[ line 0: [id1, id2, ...],
                    line 1: [id1, id2, ...],
                    ... ]
            movie 1:[ line 0: [id1, id2, ...],
                    line 1: [id1, id2, ...],
                    ... ]
            movie 2:[ line 0: [id1, id2, ...],
                    line 1: [id1, id2, ...],
                    ... ]
            ...

        ]
        
        scarce_words_counts: A list with words that only appear once
        [ token1, token2, token3, ...]
        
        """
        # Replace the word with <UNK> that appears only once.
        # for movie in all_toks_new:
        for i in range(len(all_toks_new)):
            if i % 100 == 0:
                print("Iteration (per 100 movies): ",int(i/100))
            for toks in all_toks_new[i]:
                for j in range(len(toks)):
                    if toks[j] in scarce_words_counts:
                        toks[j] = "<UNK>"
        
        # Cut the sentence to mxlen
        for movie in all_toks_new:
            for i in range(len(movie)):
                movie[i] = np.array(movie[i][:self.mxlen])
        
        return all_toks_new



class Data:

    def __init__(self, all_lineids, all_ids, all_toks_new):
        
        self.mxlen = 20

        self.input_tokens = []
        self.output_tokens = []
        self.make_data(all_lineids, all_ids, all_toks_new)

        self.input_word2id = {} # Count input word library
        self.output_word2id = {} # Count output word library
        self.input_id2word = {} # Reverse input_word2id
        self.output_id2word = {} # Reverse output_word2id
        self.encoder_input_data = np.zeros((len(self.input_tokens), self.mxlen), dtype='float32')
        self.decoder_input_data = np.zeros((len(self.output_tokens), self.mxlen), dtype='float32')
        self.num_encoder_tokens = 0
        self.num_decoder_tokens = 0
        self.seq2idx()
        
    
    def separate_conv(self, ids, toks):
        """
        Separate the sequence of characters and their words if they utter continuously without waiting for the other to speak
        For example:
        ids = [2, 0, 2, 0, 2, 0, 0 ,2]
        toks = [tok1, tok2, tok3, tok4, tok5, tok6, tok7, tok8]
        sep_toks = [[tok1, tok2, tok3, tok4, tok5, tok6], [tok7, tok8]]
        
        """
        sep_toks = []
        for i in range(len(ids)):
            if i == 0:
                temp = ids[i]
                idx = i
            else:
                if temp == ids[i]:
                    sep_toks.append(toks[idx:i])
                    idx = i
                temp = ids[i]
            
            if i == (len(ids)-1):
                sep_toks.append(toks[idx:len(ids)])
        
        return sep_toks
    
    def make_data(self, all_lineids, all_ids, all_toks_new):
        """
        Transform our original data with all dialogues all_toks_new into training data (input_tokens, output_tokens)
        
        A movie can be seen as an entity with sequential characters' conversations.
        We deem a conversation end when two line ids are not consecutive.
        for example, if a lineid sequence is [242, 241, 237, 236, 235]
        we can make it into two conversations: [242, 241], [237, 236, 235]
        
        After specifying the conversations, we can then prepare the training data as follows:
        for two conversations: [242, 241], [237, 236, 235] and corresponding token sequence is [toks1, toks2], [toks3, toks4, toks5]
        we make input_tokens as [toks2], [toks4, toks5]
                output_tokens as [toks1], [toks3, toks4]
        
        
        Then we combine all tokens input-output pairs of every conversation in every movie.
        so we will have
        input_tokens = [toks2, toks4, toks5, toks7, ...]
        output_tokens = [toks1, toks3, toks4, toks6, ...]
        
        Finally we wish to have our target output tokens to be almost same as output_tokens with each data ahead by one timestep.
        output_target_tokens = [toks1, toks3, toks4, toks6, ...]
        
        """
        
        N = len(all_lineids) #number of movies

        for i in range(N):
            #For a single movie
            movie = all_lineids[i]
            for j in range(len(movie)):
                if j == 0:
                    temp = movie[j]
                    idx = j
                else:
                    if (temp-movie[j]) is not 1:
                        sep_toks = self.separate_conv(all_ids[i][idx:j], all_toks_new[i][idx:j])
                        for toks in sep_toks:
                            self.input_tokens += toks[1:]
                            self.output_tokens += toks[:-1]

                        idx = j
                    temp = movie[j]

                #Last Sequence
                if j == len(movie)-1:
                    sep_toks = self.separate_conv(all_ids[i][idx:len(movie)], all_toks_new[i][idx:len(movie)])
                    for toks in sep_toks:
                        self.input_tokens += toks[1:]
                        self.output_tokens += toks[:-1]
        
        self.input_tokens = np.asarray(self.input_tokens)
        self.output_tokens = np.asarray(self.output_tokens)
    
    def seq2idx(self):
        # Initialize parameters
        all_input_words = set()
        all_output_words = set()

        # Calculate input words and output words as a sorted list
        for toks in self.input_tokens:
            for tok in toks:
                if tok not in all_input_words:
                    all_input_words.add(tok)
        for toks in self.output_tokens:
            for tok in toks:
                if tok not in all_output_words:
                    all_output_words.add(tok)
        all_input_words = sorted(list(all_input_words))
        all_output_words = sorted(list(all_output_words))

        # Make input and output libraries
        self.num_encoder_tokens = len(all_input_words)
        self.num_decoder_tokens = len(all_output_words)
        self.input_word2id = dict([(word, i) for i, word in enumerate(all_input_words)])
        self.output_word2id = dict([(word, i) for i, word in enumerate(all_output_words)])
        self.input_id2word = dict((i, tok) for tok, i in self.input_word2id.items())
        self.output_id2word = dict((i, tok) for tok, i in self.output_word2id.items())

        # Make encoder_input_data, decoder_input_data, decoder_target_data
        for i, (input_text, output_text) in enumerate(zip(self.input_tokens, self.output_tokens)):
            for t, word in enumerate(input_text):
                self.encoder_input_data[i, t] = self.input_word2id[word]
            for t, word in enumerate(output_text):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                self.decoder_input_data[i, t] = self.output_word2id[word]


if __name__ == '__main__':
    
    vocab = Vocabulary("movie_lines.tsv")
    all_lineids, all_ids, all_toks = vocab.load_data()
    all_toks_new, scarce_words_counts = vocab.text_tokenize(all_toks)
    
    ############################################
    # Default: Skip, load the data directly
    # all_toks_new = vocab.modify(all_toks_new, scarce_words_counts)
    # file1=open("all_toks_new.bin","wb")
    # file2=open("word2id.bin","wb")
    # pickle.dump(all_toks_new,file1)
    # pickle.dump(vocab.word2id,file2)
    # file1.close()
    # file2.close()
    ############################################

    # Load the processed data to save time
    file1=open("all_toks_new.bin","rb")
    file2=open("word2id.bin","rb")
    all_toks_new=pickle.load(file1)
    word2id=pickle.load(file2)
    
    data = Data(all_lineids, all_ids, all_toks_new)
    
    print(data.input_tokens[0])
    print(data.output_tokens[0])
    print(data.encoder_input_data[0])
    print(data.decoder_input_data[0])