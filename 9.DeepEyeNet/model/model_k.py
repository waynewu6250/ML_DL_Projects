from keras.preprocessing import sequence, image
from keras.models import Sequential, Model, Input
from keras.layers import LSTM, Embedding, Dense, Activation, Add
from keras.layers import RepeatVector, Flatten, Reshape, Concatenate, Lambda, TimeDistributed, Dot
from keras.layers.wrappers import Bidirectional
import keras.backend as K
import numpy as np
import re

##############################################################
#                        BaseModel                           #
##############################################################

class KeywordModel:
    def __init__(self, param):
        
        # Create caption model
        self.caption_model = Sequential([
                Embedding(param["vocab_size"], param["embedding_size"], input_length=param["max_len"]),
                LSTM(256, return_sequences=True),
                TimeDistributed(Dense(300))
            ])
        
        self.key_max_len = param["key_max_len"]
        self.keywords_ids = param["keywords_ids"]
        self.embedding_size = param["embedding_size"]
        self.vocab_size = param["vocab_size"]
        self.max_len = param["max_len"]
        self.word2id = param["word2id"]
        self.id2word = param["id2word"]
    
    #virtual function
    def forward(self):
       raise NotImplementedError( "forwardMethod is virutal! Must be overwrited." )

    #training generator
    def data_generator_k(self, imgs_features, imgs, cap_ids, batch_size = 32):
        def initialize():
            partial_caps = [] # Every single state of input words
            current_imgs = [] #Every image by the current state
            next_words = [] #Predicted next word
            count = 0
            return partial_caps, current_imgs, next_words, count
    
        partial_caps, current_imgs, next_words, count = initialize()
        current_keys = []
        
        while True:
            # For every image data
            for img_path in imgs:
                current_img = imgs_features[img_path]
                current_key = self.keywords_ids[img_path]
                current_key = sequence.pad_sequences(np.asarray(current_key)[np.newaxis,:], maxlen=self.key_max_len, padding='post').squeeze(0)
                
                # For every token in single caption data
                for j in range(len(cap_ids[img_path])-1):
                    count += 1
                    partial_caps.append(cap_ids[img_path][:j+1])
                    current_imgs.append(current_img)
                    current_keys.append(current_key)
                    next_words.append(np.eye(self.vocab_size)[cap_ids[img_path][j+1]])

                    if count >= batch_size:
                        partial_caps = sequence.pad_sequences(partial_caps, maxlen=self.max_len, padding='post')
                        current_imgs = np.asarray(current_imgs)
                        current_keys = np.asarray(current_keys)
                        next_words = np.asarray(next_words)
                        yield ([current_imgs, partial_caps, current_keys], next_words)

                        partial_caps, current_imgs, next_words, count = initialize()
                        current_keys = []

    #prediction
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


##############################################################
#                     1.AttentionModel                       #
##############################################################
class AttentionModel(KeywordModel):
    def __init__(self, param, img_size):
        super(AttentionModel, self).__init__(param)
        
        # Softmax calculation
        def softmax(x, axis=1):
            """
            Softmax activation function.
            """
            ndim = K.ndim(x)
            if ndim == 2:
                return K.softmax(x)
            elif ndim > 2:
                e = K.exp(x - K.max(x, axis=axis, keepdims=True))
                s = K.sum(e, axis=axis, keepdims=True)
                return e / s
            else:
                raise ValueError('Cannot apply softmax to a tensor that is 1D')
        
        self.img_size = img_size

        # Create image model
        self.image_model = Sequential([
                Dense(self.embedding_size, input_shape=(self.img_size,), activation='relu'),
                RepeatVector(self.key_max_len)
            ])
        
        # Create dense model
        self.dense_model = Sequential([
            Dense(32, activation = "tanh"),
            Dense(1, activation = "relu"),
            Activation(softmax, name='attention_weights'),
        ])
        
        
    
    def forward(self):
        #  Image Input
        x1 = Input(shape=(self.img_size,))
        img_input = self.image_model(x1)
        img_input_no_repeat = Dense(self.embedding_size, input_shape=(self.img_size,), activation='relu')(x1)

        # Caption Input
        x2 = Input(shape=(self.max_len,))
        caption_input = self.caption_model(x2)

        # Keyword Input
        x3 = Input(shape=(self.key_max_len,))
        keyword_input = Embedding(self.vocab_size, 
                                self.embedding_size, 
                                input_length=self.key_max_len)(x3)

        # Attention Mechanism
        mix_input = Concatenate(axis=-1)([img_input, keyword_input])
        scores = self.dense_model(mix_input)
        context = Dot(axes = 1)([scores, keyword_input])

        context = Add()([context, img_input_no_repeat])
        x = Add()([context, caption_input])
        x = Bidirectional(LSTM(256, return_sequences=False))(x)
        out = Dense(self.vocab_size, activation='softmax')(x)

        return Model(inputs=[x1, x2, x3], outputs=out)


##############################################################
#                     2.RNNEncoder                           #
##############################################################
class EncoderModel(KeywordModel):
    def __init__(self, param, img_size):
        super(EncoderModel, self).__init__(param)
        self.img_size = img_size
    
    def forward(self):
        x1 = Input(shape=(self.img_size,))
        x2 = Input(shape=(self.max_len,))
        x3 = Input(shape=(self.key_max_len,))

        # keyword encoder
        img_input = Dense(self.embedding_size, input_shape=(self.img_size,), activation='relu')(x1)
        img_input = Lambda(lambda x: K.expand_dims(x, axis=1))(img_input)
        caption_input = self.caption_model(x2)

        keyword_input = Embedding(self.vocab_size, self.embedding_size, input_length=self.key_max_len)(x3)
        mix_input = Concatenate(axis=1)([img_input, keyword_input])
        context = LSTM(300, return_sequences=False)(mix_input)
        context = RepeatVector(self.max_len)(context)

        x = Add()([context, caption_input])
        x = Bidirectional(LSTM(256, return_sequences=False))(x)
        out = Dense(self.vocab_size, activation='softmax')(x)

        return Model(inputs=[x1, x2, x3], outputs=out)


##############################################################
#                     3.MeanEncoder                          #
##############################################################
class MeanModel(KeywordModel):
    def __init__(self, param, img_size):
        super(MeanModel, self).__init__(param)
        self.img_size = img_size
        
        # create image model
        self.image_model = Sequential([
                Dense(self.embedding_size, input_shape=(self.img_size,), activation='relu'),
                RepeatVector(self.max_len)
                ])
        
        # create keyword model
        self.keyword_model = Sequential([
            Embedding(self.vocab_size, self.embedding_size, input_length=self.key_max_len),
            Lambda(lambda x: K.mean(x, axis=1, keepdims=False)),
            RepeatVector(self.max_len)
        ])
    
    def forward(self):
        x1 = Input(shape=(self.img_size,))
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
    





