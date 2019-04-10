import json
import pickle
import re
from nltk.translate.bleu_score import sentence_bleu
import numpy as np
from tqdm import tqdm
from PIL import Image

from keras.applications import inception_v3, vgg16, vgg19, resnet50
from keras.preprocessing import image
from keras.models import Model


# 1. Load json files
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

# 2. For keyword model text preparation
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


# 3. Extract image features (inception_v3, vgg16, vgg19)
class Extract:
    def __init__(self, model, dim):
        if model not in ['inception_v3', 'vgg16', 'vgg19', 'resnet50']:
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
        elif model == 'resnet50':
            self.model = self.preload(resnet50.ResNet50(weights='imagenet'))
            self.preprocess_input = resnet50.preprocess_input
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

    
    
# 4. Evaluate Captions for bleu score
def predict_caption_bleu(img_id, imgs, imgs_feats, model, obj, results, key_flag, bs_flag):
    if key_flag:
        if bs_flag:
            predicted = text_prepare(obj.predict_captions_beam_search_k(imgs[img_id], imgs_feats, model))
        else:
            predicted = text_prepare(obj.predict_captions_k(imgs[img_id], imgs_feats, model))
    else:
        if bs_flag:
            predicted = text_prepare(obj.predict_captions_beam_search(imgs[img_id], imgs_feats, model))
        else:
            predicted = text_prepare(obj.predict_captions(imgs[img_id], imgs_feats, model))
            
    ground_truth = text_prepare(results['descriptions'][imgs[img_id]])

    # Show results
    print('Predicted Caption: {}'.format(predicted))
    print('Ground Truth Caption: {}'.format(ground_truth))

    # Bleu Score Calculation
    predicted_tok = predicted.split(" ")
    ground_truth_tok = [ground_truth.split(" ")]
    print('Bleu Score 1: {}'.format(sentence_bleu(ground_truth_tok, predicted_tok, weights=(1, 0, 0, 0))))
    print('Bleu Score 2: {}'.format(sentence_bleu(ground_truth_tok, predicted_tok, weights=(0, 1, 0, 0))))
    print('Bleu Score 3: {}'.format(sentence_bleu(ground_truth_tok, predicted_tok, weights=(0, 0, 1, 0))))
    print('Bleu Score 4: {}'.format(sentence_bleu(ground_truth_tok, predicted_tok, weights=(0, 0, 0, 1))))

    im = Image.open(imgs[img_id]+'.jpg')
    return im
    
    

# 5. Evaluate Captions for CIDEr, Rouge, Bleu score
def predict_captions(imgs, imgs_feats, model, obj, results, key_flag, bs_flag):
    # for bleu score
    avgscore1, avgscore2, avgscore3, avgscore4 = 0,0,0,0
    # for cider, rouge
    gts, res = {}, {}
    
    for i,img in tqdm(enumerate(imgs)):
        if key_flag:
            if bs_flag:
                predicted = text_prepare(obj.predict_captions_beam_search_k(img, imgs_feats, model))
            else:
                predicted = text_prepare(obj.predict_captions_k(img, imgs_feats, model))
        else:
            if bs_flag:
                predicted = text_prepare(obj.predict_captions_beam_search(img, imgs_feats, model))
            else:
                predicted = text_prepare(obj.predict_captions(img, imgs_feats, model))
            
        ground_truth = text_prepare(results['descriptions'][img])
        
        # for bleu score
        predicted_tok = predicted.split(" ")
        ground_truth_tok = [ground_truth.split(" ")]
        avgscore1 += sentence_bleu(ground_truth_tok, predicted_tok, weights=(1, 0, 0, 0))
        avgscore2 += sentence_bleu(ground_truth_tok, predicted_tok, weights=(0, 1, 0, 0))
        avgscore3 += sentence_bleu(ground_truth_tok, predicted_tok, weights=(0, 0, 1, 0))
        avgscore4 += sentence_bleu(ground_truth_tok, predicted_tok, weights=(0, 0, 0, 1))
            
        # for cider, rouge
        if img not in gts:
            gts[img] = [ground_truth]
        else:
            gts[img] = [gts[img], ground_truth]
        if img not in res:
            res[img] = [predicted]
        else:
            res[img] = [res[img], predicted]
    
    avgscore = np.asarray([avgscore1,avgscore2,avgscore3,avgscore4])
    avgscore = avgscore/len(imgs)
        
    return gts, res, avgscore


# 6. Evaluate Captions for CIDEr, Rouge
def calc_scores(scorer, name, pred_results):
    
    train_gts = pred_results['train_gts']
    train_res = pred_results['train_res']
    test_gts = pred_results['test_gts']
    test_res = pred_results['test_res']
    train_gts_k = pred_results['train_gts_k']
    train_res_k = pred_results['train_gts_k']
    test_gts_k = pred_results['test_gts_k']
    test_res_k = pred_results['test_res_k']
    
    print('-----------------------------')
    print(name+':')
    (score, scores) = scorer.compute_score(train_gts, train_res)
    print('train %s score = %.4f' % (name,score))
    (score, scores) = scorer.compute_score(test_gts, test_res)
    print('test %s score = %.4f' % (name,score))
    (score, scores) = scorer.compute_score(train_gts_k, train_res_k)
    print('train %s score (key model) = %.4f' % (name,score))
    (score, scores) = scorer.compute_score(test_gts_k, test_res_k)
    print('test %s score (key model) = %.4f' % (name,score))


# 7. Evaluate Captions for CIDEr, Rouge in keyword model
def calc_scores_k(scorer, name, names, pred_results):
    
    print('\n-----------------------------')
    print('Evaluation: ', name)
    
    for i in range(len(names)):
        test_gts = pred_results['test_gts_list'][i]
        test_res = pred_results['test_res_list'][i]
                           
        print('-----------------------------')
        print(names[i]+':')
        (score, scores) = scorer.compute_score(test_gts, test_res)
        print('test %s score = %.4f' % (names[i],score))





