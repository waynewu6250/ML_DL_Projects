import torch as t
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
import pickle
from eval_model.config import opt

class CaptionData(Dataset):

    def __init__(self, data_path, opt):
        self.raw_data = self.load_pickle(data_path)
        self.cap_toks = self.raw_data["cap_toks"]
        self.cap_ids = self.raw_data["cap_ids"]
        self.word2id = self.raw_data["word2id"]
        self.id2word = self.raw_data["id2word"]

        self.cap_labels = [(idd, img.split('/')[2]) for img, idd in self.cap_ids.items()]
        self.data, self.vocab_list = self.get_vocab(opt)
        self.pad = len(self.word2id) #Last token
        self.num_classes = len(self.vocab_list)

    def __getitem__(self, index):
        return self.data[index][0], self.data[index][1], index
    
    def __len__(self):
        return len(self.data)
        
    # Load pickle files
    def load_pickle(self, file):
        with open(file,'rb') as f:
            return pickle.load(f)
    
    def save_pickle(self, data, file):
        with open(file,'wb') as f:
            pickle.dump(data, f)
    
    # Get label vocabulary list
    def get_vocab(self, opt):
        if opt.vocab_path:
            vocab_list = t.load(opt.vocab_path)
        else:
            vocab_list = {}
            for i,label in enumerate(list(set(map(lambda x: x[1],self.cap_labels)))):
                vocab_list[label] = i
            t.save(vocab_list, 'vocab_list.pth')

        return list(map(lambda x: (x[0], vocab_list[x[1]]), self.cap_labels)), vocab_list


def get_collate_fn(pad, num_classes, max_length=50):
    def collate_fn(data):
        # Sort by caption lengths
        data = sorted(data, key = lambda x: len(x[0]), reverse=True)
        captions, labels, indexes = zip(*data)
        
        # Calculate caption length
        lengths = [min(len(c), max_length) for c in captions]
        batch_length = max(lengths)

        # Captions into tensor
        captions_t = t.LongTensor(batch_length, len(captions)).fill_(pad)
        labels_t = t.LongTensor(np.asarray(labels))
        for i,cap in enumerate(captions):
            end_cap = lengths[i]
            captions_t[:end_cap,i].copy_(t.LongTensor(cap))
        return (captions_t, lengths), labels_t, indexes
    return collate_fn

def get_dataloader(opt):
    dataset = CaptionData(opt.data_path, opt)
    return DataLoader(dataset, 
                      batch_size=opt.batch_size, 
                      shuffle=False, 
                      collate_fn=get_collate_fn(dataset.pad,dataset.num_classes))


