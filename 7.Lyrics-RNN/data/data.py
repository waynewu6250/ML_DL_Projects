import numpy as np
import os
import re

def parse(filename):
    
    f = open(filename)
    data = []
    lyrics = ''
    
    for line in f:
        
        if line.find('監') != -1 or line.find('Repeat') != -1 or line.find('O.S.') != -1:
            continue
        if line == 'end\n' or line.find('-') != -1:
            if lyrics is not '':
                data.append(lyrics[:-1])
            lyrics = ''
            continue
        # Case 1: remove (周) or 周: 
        line = re.sub(r'^\(.*?\)',' ',line)
        line = re.sub(r'.*：','',line)
        # Case 2: remove some redundant tokens
        line = re.sub(r'(\()|(\))|[\'şı,＊＃☆△…＠í대]','',line)
        # Case 3: remove english character
        line = re.sub(r'[a-zA-Z]','',line)
        line = line.strip(" ")
        if line == '\n':
            continue

        lyrics += line
    
    return data

def padding(sequences, idd):
    
    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    
    lengths = []
    for s in sequences:
        if not hasattr(s, '__len__'):
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(s))
        lengths.append(len(s))
    
    num_samples = len(sequences)
    mxlen = np.max(lengths)
    encoded_input_data = (np.ones((num_samples,mxlen)) * idd).astype('int32')

    for (idx,s) in enumerate(sequences):
        encoded_input_data[idx,:len(s)] = s
    
    return encoded_input_data


def get_data(opt):
    
    if os.path.exists(opt.pickle_path):
        data = np.load(opt.pickle_path)
        data, word2ix, ix2word = data['data'], data['word2ix'].item(), data['ix2word'].item()
        return data, word2ix, ix2word
    
    #==============================================================#
    # Parse the data
    data = parse(opt.filename)
    # pop the one with too long lyrics
    delete = [23,38,255]
    for i in delete:
        data.pop(i)

    # lengths = [len(x) for x in data]
    # print(np.argmax(lengths))
    # print(np.max(lengths))
    # print(data[np.argmax(lengths)])

    # Make dictionaries
    words = {_word for _sentence in data for _word in _sentence}
    word2ix = {_word: _ix for _ix, _word in enumerate(words)}
    word2ix['<EOS>'] = len(word2ix)  # End token
    word2ix['<START>'] = len(word2ix)  # Start token
    word2ix['</s>'] = len(word2ix)  # Space token
    ix2word = {_ix: _word for _word, _ix in list(word2ix.items())}

    # Make tokens
    for i in range(len(data)):
        data[i] = ["<START>"] + list(data[i]) + ["<EOS>"]
    
    new_data = [[word2ix[_word] for _word in _sentence]
                for _sentence in data]
    
    # Make encoded data
    pad_data = padding(new_data, word2ix['</s>'])
    np.savez_compressed(opt.pickle_path,
                        data = pad_data,
                        word2ix = word2ix,
                        ix2word = ix2word)
    
    return pad_data, word2ix, ix2word


