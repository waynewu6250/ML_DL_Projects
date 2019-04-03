import numpy as np
import re
import math
import pickle
import h5py

from keras.models import Sequential
from keras.models import Model, load_model
from keras.layers import LSTM, Embedding 
from keras.layers import Input
from keras.layers import Dense, Flatten, Reshape
from keras import optimizers

# hyperparameter
mxlen = 20          # Max length for a sequence of tokens
batch_size = 32    # Batch size for training.
epochs = 20         # Number of epochs to train for.
char_dim=  64       # Embedding size
latent_dim = 192     # Latent dimensionality of the encoding space.

# dictionary
word2id = {}        # Count all word library
id2word = {}        # Reverse word2id
input_word2id = {}  # Count input word library
output_word2id = {} # Count output word library
input_id2word = {}  # Reverse input_word2id
output_id2word = {} # Reverse output_word2id

data_path = "movie_lines.tsv"

#===================================================#
#===================================================#
# Functions:

def load_data(filename):
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
    file = open(filename)
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

#======================#
#1. Text Prepare
# 1) Remove bad symbols and tokenization
REPLACE_BY_SPACE_RE = re.compile(r'[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile(r'[^0-9a-z #+_]')
def text_prepare(text):
    """
        text: a string
        
        return: modified string tokens 
                [tok1, tok2 , ...] which is a single sentence from one character
    """
    tok = ["<START>"] # add START token to represent sentence start
    text = text.lower() # lowercase text
    text = re.sub(REPLACE_BY_SPACE_RE, ' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(BAD_SYMBOLS_RE, '', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    tok += (text.split()+["<EOS>"]) # add EOS token to represent sentence end
    
    return tok

# 2) Dictionary of all words from train corpus with their counts.
#    Dictionary of all words with its ids
def count_words(all_toks):
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
    word2id["<START>"] = 0
    word2id["<EOS>"] = 1
    word2id["<UNK>"] = 2
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
                if not word in word2id:
                    word2id[word] = index
                    index += 1
    
    return count
def text_tokenize(all_toks, word2id, id2word):
    
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
        toks = [text_prepare(x) for x in toks]
        all_toks_new.append(toks)

    # Count the words that appears only once.
    words_counts = count_words(all_toks_new)
    scarce_words_counts = [x[0] for x in sorted(words_counts.items(), key = lambda x: x[1], reverse=True) if x[1] == 1]
    
    # Remove scarce words in word2id dictionary and reindex all words
    for word in scarce_words_counts:
        del word2id[word]
    
    # Arrange word2id and id2word
    word2id = {key: i for i, key in enumerate(word2id.keys())}
    id2word = {i:symbol for symbol, i in word2id.items()}
    
    return all_toks_new, scarce_words_counts, word2id, id2word
#======================#
#2. Replace and Restrict Word Length
def modify(all_toks_new, scarce_words_counts, mxlen):
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
            movie[i] = np.array(movie[i][:mxlen])
    
    return all_toks_new
#======================#
#3. Making encoding and decoding data
def separate_conv(ids, toks):
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
def make_data(all_lineids, all_ids, all_toks_new):
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
    input_tokens = []
    output_tokens = []
    
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
                    sep_toks = separate_conv(all_ids[i][idx:j], all_toks_new[i][idx:j])
                    for toks in sep_toks:
                        input_tokens += toks[1:]
                        output_tokens += toks[:-1]

                    idx = j
                temp = movie[j]

            #Last Sequence
            if j == len(movie)-1:
                sep_toks = separate_conv(all_ids[i][idx:len(movie)], all_toks_new[i][idx:len(movie)])
                for toks in sep_toks:
                    input_tokens += toks[1:]
                    output_tokens += toks[:-1]
            
    return input_tokens, output_tokens



#===================================================#
#===================================================#
# Load data
all_lineids, all_ids, all_toks = load_data(data_path)
# Text Prepare
all_toks_new, scarce_words_counts, word2id, id2word = text_tokenize(all_toks, word2id, id2word)

# Replace and restict word length
# all_toks_new = modify(all_toks_new, scarce_words_counts, mxlen)
# file1=open("all_toks_new.bin","wb")
# file2=open("word2id.bin","wb")
# pickle.dump(all_toks_new,file1)
# pickle.dump(word2id,file2)
# file1.close()
# file2.close()

# Load the processed data to save time
file1=open("all_toks_new.bin","rb")
file2=open("word2id.bin","rb")
all_toks_new=pickle.load(file1)
word2id=pickle.load(file2)

# Make Tokens
input_tokens, output_tokens = make_data(all_lineids, all_ids, all_toks_new)
input_tokens = np.asarray(input_tokens)
output_tokens = np.asarray(output_tokens)
# print(input_tokens[0])
# print(output_tokens[0])
# print(len(input_tokens))

input_tokens = input_tokens[:30000]
output_tokens = output_tokens[:30000]

# Initialize parameters
all_input_words = set()
all_output_words = set()

# Calculate input words and output words as a sorted list
for toks in input_tokens:
    for tok in toks:
        if tok not in all_input_words:
            all_input_words.add(tok)
for toks in output_tokens:
    for tok in toks:
        if tok not in all_output_words:
            all_output_words.add(tok)
all_input_words = sorted(list(all_input_words))
all_output_words = sorted(list(all_output_words))

# Make input and output libraries
num_encoder_tokens = len(all_input_words)
num_decoder_tokens = len(all_output_words)
input_word2id = dict([(word, i) for i, word in enumerate(all_input_words)])
output_word2id = dict([(word, i) for i, word in enumerate(all_output_words)])
input_id2word = dict((i, tok) for tok, i in input_word2id.items())
output_id2word = dict((i, tok) for tok, i in output_word2id.items())

# Make encoder_input_data, decoder_input_data, decoder_target_data
encoder_input_data = np.zeros((len(input_tokens), mxlen), dtype='float32')
decoder_input_data = np.zeros((len(output_tokens), mxlen), dtype='float32')
decoder_target_data = np.zeros((len(output_tokens), mxlen, num_decoder_tokens), dtype='float32')

for i, (input_text, output_text) in enumerate(zip(input_tokens, output_tokens)):
    for t, word in enumerate(input_text):
        encoder_input_data[i, t] = input_word2id[word]
    for t, word in enumerate(output_text):
        # decoder_target_data is ahead of decoder_input_data by one timestep
        decoder_input_data[i, t] = output_word2id[word]
        if t > 0:
            # decoder_target_data will be ahead by one timestep
            # and will not include the start character.
            decoder_target_data[i, t - 1, output_word2id[word]] = 1.


#===================================================#
#===================================================#
e_model = load_model("encoder.h5")
d_model = load_model("decoder.h5")

def decode_sequence(input_seq,e_model,d_model):
    
    # Encode the input as state vectors.
    states_value = e_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1,1))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0] = output_word2id['<START>']

    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        outputs, h, c = d_model.predict(
            [target_seq] + states_value)
        
        # Sample a token
        sampled_token_index = np.argmax(outputs[0, -1, :])
        sampled_tok = output_id2word[sampled_token_index]
        decoded_sentence += ' '+sampled_tok

        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_tok == '<EOS>'):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_seq = np.zeros((1,1))
        target_seq[0, 0] = sampled_token_index

        # Update states
        states_value = [h, c]
    
    decoded_sentence = decoded_sentence.strip('<EOS>')
    if decoded_sentence == ' ':
        decoded_sentence = 'I have nothing to say'

    return decoded_sentence

def tokenize_seq(data):
    token_data = [text_prepare(text) for text in data]
    encoder_data = np.zeros((len(token_data), mxlen), dtype='float32')

    for i, input_text in enumerate(token_data):
        for t, word in enumerate(input_text):
            if word in input_word2id:
                encoder_data[i, t] = input_word2id[word]
            else:
                encoder_data[i, t] = 2
    return encoder_data

# Test input data
# data = ["my name is david, what is my name?",
#         "my name is john, what is my name?",
#         "are you a leader or a follower?",
#         "are you a follower or a leader?",
#         "what is moral?",
#         "what is immoral?",
#         "what is altruism?",
#         "ok ... so what is the deﬁnition of morality?",
#         "tell me the deﬁnition of morality , i am quite upset now!"]

# encoder_data = tokenize_seq(data)
# for seq_index in range(len(encoder_data)):
#     input_seq = encoder_data[seq_index:seq_index+1]
#     decoded_sentence = decode_sequence(input_seq,e_model,d_model)
#     print('-')
#     print('Input sentence:', data[seq_index])
#     print('Decoded sentence:', decoded_sentence)

# # Randomly test data in training set
# for seq_index in np.random.permutation(len(encoder_input_data))[:100]:
#     input_seq = encoder_input_data[seq_index:seq_index+1]
#     decoded_sentence = decode_sequence(input_seq,e_model,d_model)
#     print('-')
#     print('Input sentence:', " ".join(input_tokens[seq_index][1:-1].tolist()))
#     print('Decoded sentence:', decoded_sentence)

first = True
while True:
    if first:
        data = input("Please input your words: ")
        first = False
    else:
        data = input("You: ")
    encoder_data = tokenize_seq(data)
    input_seq = encoder_data[0:1]
    decoded_sentence = decode_sequence(input_seq,e_model,d_model)
    print("WayneBot: ",decoded_sentence)

