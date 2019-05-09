import keras.backend as K
from keras.optimizers import RMSprop
from keras.models import load_model
import h5py
import os

from model.model import CaptionModel
from model.model_k import EncoderModel, MeanModel, TransformerModel
from utils import *
from eval_tools import Cider, Rouge

from argparse import ArgumentParser

class TrainModel:
    
    def __init__(self, model_name, select_model):
        self.model_name = model_name
        self.select_model = select_model
        
        # images and image features
        self.train_imgs = load_pickle('./data/data_path/train_imgs.pkl')
        self.val_imgs = load_pickle('./data/data_path/val_imgs.pkl')
        self.test_imgs = load_pickle('./data/data_path/test_imgs.pkl')
        self.train_imgs_features = load_pickle('./data/img_features/train_imgs_features_{}_sub.pkl'.format(select_model))
        self.val_imgs_features = load_pickle('./data/img_features/val_imgs_features_{}_sub.pkl'.format(select_model))
        self.test_imgs_features = load_pickle('./data/img_features/test_imgs_features_{}_sub.pkl'.format(select_model))
        
        self.feat_sizes = {'vgg16': 4096,
                           'vgg19': 4096,
                           'inceptionV3': 2048, 
                           'resnet50': 2048}
        
        # important features
        self.results = load_pickle('./data/data_path/results.pkl')

        self.keywords = self.results["keywords"]
        self.descriptions = self.results["descriptions"]
        self.cap_toks = self.results["cap_toks"]
        self.cap_ids = self.results["cap_ids"]
        self.word2id = self.results["word2id"]
        self.id2word = self.results["id2word"]
        self.word2id_keys = self.results["word2id_keys"]
        self.keywords_ids = self.results["keywords_ids"]
        self.max_len = max([len(x) for x in self.cap_toks.values()])
        
        # Parameters
        self.vocab_size = len(self.word2id)
        self.steps_per_epoch = sum([len(self.cap_toks[img])-1 for img in self.train_imgs])
        self.val_steps = sum([len(self.cap_toks[img])-1 for img in self.val_imgs])
        self.batch_size = 32
        
        self.train_seq = ('train_gts_list', 'train_res_list', 'train_avgscore_list')
        self.test_seq = ('test_gts_list', 'test_res_list', 'test_avgscore_list')
        
        # Load Model
        # Plain model
        if self.model_name == "normal":
            self.model_obj = CaptionModel(300, self.vocab_size, self.max_len, self.word2id, self.id2word, self.feat_sizes[self.select_model])
            self.final_model = self.model_obj.forward()
            self.model_path = './checkpoints/model_{}.h5'.format(self.select_model)
        
        # 1. RNN-Encoder for keyword embedding
        if self.model_name == "encoder":
            self.model_obj = EncoderModel(self.set_params(), self.feat_sizes[self.select_model])
            self.final_model = self.model_obj.forward()
            self.model_path = './checkpoints/model_{}_keywords_encoder.h5'.format(self.select_model)

        # 2. RNN-Encoder for keyword embedding
        elif self.model_name == "mean":
            self.model_obj = MeanModel(self.set_params(), self.feat_sizes[self.select_model])
            self.final_model = self.model_obj.forward()
            self.model_path = './checkpoints/model_{}_keywords_mean.h5'.format(self.select_model)

        # 3. Self-attention-Encoder for keyword embedding
        elif self.model_name == "transformer":
            self.model_obj = TransformerModel(self.set_params(), self.feat_sizes[self.select_model], 64)
            self.final_model = self.model_obj.forward()
            self.model_path = './checkpoints/model_{}_keywords_transformer.h5'.format(self.select_model)
   
    def set_params(self):
        
        param = {}
        param["key_max_len"] = max([len(x) for x in self.keywords_ids.values()])
        param["keywords_ids"] = self.results['keywords_ids']
        param["embedding_size"] = 300
        param["vocab_size"] = len(self.word2id_keys)
        param["max_len"] = max([len(x) for x in self.cap_toks.values()])
        param["word2id"] = self.results["word2id"]
        param["id2word"] = self.results["id2word"]
        
        return param
    
    def train(self, load=True, train=True):
        if load and os.path.exists(self.model_path):
            if self.model_name == "transformer":
                self.final_model.load_weights(self.model_path)
            else:
                self.final_model = load_model(self.model_path)
            if not train:
                return self.final_model
        
        self.final_model.compile(loss='categorical_crossentropy', optimizer=RMSprop(), metrics=['accuracy'])
        
        if self.model_name == "normal":
            self.final_model.fit_generator(self.model_obj.data_generator(self.max_len, self.train_imgs_features, self.train_imgs, \
                                                                         self.cap_ids, batch_size=self.batch_size), 
                                           steps_per_epoch=self.steps_per_epoch, 
                                           validation_data = self.model_obj.data_generator(self.max_len, self.val_imgs_features, \
                                                                                           self.val_imgs, \
                                                                                           self.cap_ids, \
                                                                                           batch_size = self.batch_size),
                                           validation_steps = self.val_steps,
                                           epochs=1)
        else:
            self.final_model.fit_generator(self.model_obj.data_generator_k(self.train_imgs_features, self.train_imgs, \
                                                                        self.cap_ids, batch_size = self.batch_size), 
                                           steps_per_epoch=self.steps_per_epoch, 
                                           validation_data = self.model_obj.data_generator_k(self.val_imgs_features, \
                                                                                            self.val_imgs, \
                                                                                            self.cap_ids, \
                                                                                            batch_size = self.batch_size),
                                           validation_steps = self.val_steps,
                                           epochs=1)
        
        if self.model_name == "transformer":
            self.final_model.save_weights(self.model_path)
        else:
            self.final_model.save(self.model_path)

        return self.final_model
    
if __name__ == "__main__":
    
    parser = ArgumentParser()
    parser.add_argument("-n", "--model-name", help="model name to train", dest="model_name", default="encoder")
    parser.add_argument("-f", "--feature", help="image feature to use", dest="select_model", default="vgg16")
    args = parser.parse_args()
    
    model_manager = TrainModel(args.model_name, args.select_model)
    final_model = model_manager.train(True, True)
    

    