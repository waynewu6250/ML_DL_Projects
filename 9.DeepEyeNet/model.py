from keras.preprocessing import sequence, image
from keras.models import Sequential, Model, Input
from keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector, Activation, Flatten, Add, Lambda
from keras.layers.wrappers import Bidirectional
import keras.backend as K
import numpy as np
import re

class CaptionModel:
    def __init__(self, embedding_size, vocab_size, max_len, word2id, id2word):
        # Create image model
        self.image_model = Sequential([
                Dense(embedding_size, input_shape=(4096,), activation='relu'),
                RepeatVector(max_len)
                ])
        # Create caption model
        self.caption_model = Sequential([
                Embedding(vocab_size, embedding_size, input_length=max_len),
                LSTM(256, return_sequences=True),
                TimeDistributed(Dense(300))
            ])
        
        self.embedding_size = embedding_size
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.word2id = word2id
        self.id2word = id2word
    
    def forward(self):
        x1 = Input(shape=(4096,))
        x2 = Input(shape=(self.max_len,))
        img_input = self.image_model(x1)
        caption_input = self.caption_model(x2)
        x = Add()([img_input, caption_input])
        x = Bidirectional(LSTM(256, return_sequences=False))(x)
        out = Dense(self.vocab_size, activation='softmax')(x)

        return Model(inputs=[x1, x2], outputs=out)
    
    def predict_captions(self, image, images_features, model):
        start_word = ["<START>"]
        while True:
            par_caps = [self.word2id[i] for i in start_word]
            par_caps = sequence.pad_sequences([par_caps], maxlen=self.max_len, padding='post')
            e = images_features[image]

            preds = model.predict([np.array([e]), np.array(par_caps)])
            word_pred = self.id2word[np.argmax(preds[0])]
            start_word.append(word_pred)

            if word_pred == "<EOS>" or len(start_word) > self.max_len:
                break

        return ' '.join(start_word[1:-1])
    
    def predict_captions_beam_search(self, image, images_features, model, beam_index = 3):
        start = [self.word2id["<START>"]]

        start_word = [[start, 0.0]]

        while len(start_word[0][0]) < self.max_len:
            temp = []
            for s in start_word:
                par_caps = sequence.pad_sequences([s[0]], maxlen=self.max_len, padding='post')
                e = images_features[image[:]]

                preds = model.predict([np.array([e]), np.array(par_caps)])
                word_preds = np.argsort(preds[0])[-beam_index:]

                # Getting the top <beam_index>(n) predictions and creating a 
                # new list so as to put them via the model again
                for w in word_preds:
                    next_cap, prob = s[0][:], s[1]
                    next_cap.append(w)
                    prob += preds[0][w]
                    temp.append([next_cap, prob])

            start_word = temp
            # Sorting according to the probabilities
            start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
            # Getting the top words
            start_word = start_word[-beam_index:]

        start_word = start_word[-1][0]
        intermediate_caption = [self.id2word[i] for i in start_word]

        final_caption = []

        for i in intermediate_caption:
            if i != '<EOS>':
                final_caption.append(i)
            else:
                break

        final_caption = ' '.join(final_caption[1:])
        return final_caption
    
    # For tokenizaiton
    @staticmethod
    def text_prepare(text):
        """
            text: a string

            return: modified string tokens 
                    [tok1, tok2 , ...] which is a single sentence from one character
        """
        REPLACE_BY_SPACE_RE = re.compile('[-(){}\[\]\|@;]')
        BAD_SYMBOLS_RE = re.compile('[#+_]')
        mxlen = 50

        tok = ["<START>"] # add START token to represent sentence start
        text = text.lower() # lowercase text
        text = re.sub(REPLACE_BY_SPACE_RE, ' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
        text = re.sub(BAD_SYMBOLS_RE, '', text) # delete symbols which are in BAD_SYMBOLS_RE from text
        tok += (text.split(" ")+["<EOS>"]) # add EOS token to represent sentence end
        if len(tok) > mxlen:
            tok = tok[:mxlen]

        return tok
    
    
class KeywordModel(CaptionModel):
    
    def __init__(self, key_max_len, keywords_ids, embedding_size, new_vocab_size, vocab_size, max_len, word2id, id2word):
        super(KeywordModel, self).__init__(embedding_size, vocab_size, max_len, word2id, id2word)
        
        # create keyword model
        self.keyword_model = Sequential([
            Embedding(new_vocab_size, self.embedding_size, input_length=key_max_len),
            Lambda(lambda x: K.mean(x, axis=1, keepdims=False)),
            RepeatVector(self.max_len)
        ])
        
        self.key_max_len = key_max_len
        self.keywords_ids = keywords_ids
        self.vocab_size = new_vocab_size
     
    def forward(self):
        x1 = Input(shape=(4096,))
        x2 = Input(shape=(self.max_len,))
        x3 = Input(shape=(self.key_max_len,))

        img_input = self.image_model(x1)
        caption_input = self.caption_model(x2)
        keyword_input = self.keyword_model(x3)

        x = Add()([img_input, keyword_input])
        x = Add()([x, caption_input])
        x = Bidirectional(LSTM(256, return_sequences=False))(x)
        out = Dense(self.vocab_size, activation='softmax')(x)

        return Model(inputs=[x1, x2, x3], outputs=out)
    
    def predict_captions_k(self, image, images_features, model):
        start_word = ["<START>"]
        while True:
            par_caps = [self.word2id[i] for i in start_word]
            par_caps = sequence.pad_sequences([par_caps], maxlen=self.max_len, padding='post')
            current_key = self.keywords_ids[image]
            current_key = sequence.pad_sequences(np.asarray(current_key)[np.newaxis,:],
                                                 maxlen=self.key_max_len, padding='post').squeeze(0)
            e = images_features[image]

            preds = model.predict([np.array([e]), np.array(par_caps), np.array([current_key])])
            word_pred = self.id2word[np.argmax(preds[0])]
            start_word.append(word_pred)

            if word_pred == "<EOS>" or len(start_word) > self.max_len:
                break

        return ' '.join(start_word[1:-1])
    
    def predict_captions_beam_search_k(self, image, images_features, model, beam_index = 3):
        start = [self.word2id["<START>"]]

        start_word = [[start, 0.0]]

        while len(start_word[0][0]) < self.max_len:
            temp = []
            for s in start_word:
                par_caps = sequence.pad_sequences([s[0]], maxlen=self.max_len, padding='post')
                current_key = self.keywords_ids[image]
                current_key = sequence.pad_sequences(np.asarray(current_key)[np.newaxis,:],
                                                     maxlen=self.key_max_len, padding='post').squeeze(0)
                e = images_features[image[:]]

                preds = model.predict([np.array([e]), np.array(par_caps), np.array([current_key])])
                word_preds = np.argsort(preds[0])[-beam_index:]

                # Getting the top <beam_index>(n) predictions and creating a 
                # new list so as to put them via the model again
                for w in word_preds:
                    next_cap, prob = s[0][:], s[1]
                    next_cap.append(w)
                    prob += preds[0][w]
                    temp.append([next_cap, prob])

            start_word = temp
            # Sorting according to the probabilities
            start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
            # Getting the top words
            start_word = start_word[-beam_index:]

        start_word = start_word[-1][0]
        intermediate_caption = [self.id2word[i] for i in start_word]

        final_caption = []

        for i in intermediate_caption:
            if i != '<EOS>':
                final_caption.append(i)
            else:
                break

        final_caption = ' '.join(final_caption[1:])
        return final_caption
        
        
    
        