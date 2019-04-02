import json
import pickle
import re
import numpy as np
from tqdm import tqdm

from keras.applications import inception_v3, vgg16, vgg19 
from keras.preprocessing import image
from keras.models import Model


# Load json files
def load_json(file):
    with open(file,'r') as f:
        return json.load(f)
def save_json(data, file):
    with open(file,'w') as f:
        json.dump(data, f)
def load_pickle(file):
    with open(file,'rb') as f:
        return pickle.load(f)
def save_pickle(data, file):
    with open(file,'wb') as f:
        pickle.dump(data, f)

# For keyword model text preparation
def text_prepare(text):
    """
        text: a string

        return: modified string tokens 
                [tok1, tok2 , ...] which is a single sentence from one character
    """
    REPLACE_BY_SPACE_RE = re.compile('[-(){}\[\]\|@;]')
    BAD_SYMBOLS_RE = re.compile('[#+_]')
    mxlen = 50

    tok = []
    text = text.lower()
    text = re.sub(REPLACE_BY_SPACE_RE, ' ', text)
    text = re.sub(BAD_SYMBOLS_RE, '', text)
    tok += text.split(" ")
    if len(tok) > mxlen:
        tok = tok[:mxlen]

    return " ".join(tok)


# Extract image features
class Extract:
	def __init__(self, model, dim):
		if model not in ['inception_v3', 'vgg16', 'vgg19']:
			raise ValueError('Please specify correct model!!')
		if model == 'inception_v3':
			self.model = self.preload(inception_v3.InceptionV3(weights='imagenet'))
			self.preprocess_input = inception_v3.preprocess_input
		elif model == 'vgg16':
			self.model = self.preload(vgg16.VGG16(weights='imagenet'))
			self.preprocess_input = vgg16.preprocess_input
		elif model == 'vgg19':
			self.model = self.preload(vgg19.VGG19(weights='imagenet'))
			self.preprocess_input = vgg19.preprocess_input
		self.dim = dim
	
	# Preprocess the images
	def preprocess_img(self, img_path):
		img = image.load_img(img_path, target_size=(self.dim, self.dim))
		x = image.img_to_array(img)
		x = np.expand_dims(x, axis=0)
		x = self.preprocess_input(x)
		return x
	
	def preload(self, model):
		new_input = model.input
		hidden_layer = model.layers[-2].output
		return Model(new_input, hidden_layer)
	
	def extract_features(self, imgs):
		imgs_features = {}
		for img in tqdm(imgs):
			features = self.model.predict(self.preprocess_img(img+'.jpg'))
			imgs_features[img] = features.squeeze()
		return imgs_features
	



