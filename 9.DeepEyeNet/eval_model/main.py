import torch as t
from torch.autograd import Variable
from torch.optim import Adam
import numpy as np
import tqdm
from data import get_dataloader
from class_model import ClrModel
from config import opt

def train(**kwargs):

    #Parameters
    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    for k, v in kwargs.items():
        setattr(opt, k, v)

    #Data
    dataloader = get_dataloader(opt)

    #Model
    model = ClrModel(dataloader.dataset.word2id, dataloader.dataset.num_classes, opt)
    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path, map_location='cpu'))
    model = model.to(device)

    optimizer = Adam(model.parameters(), opt.lr)
    criterion = t.nn.CrossEntropyLoss()
    for epoch in range(opt.max_epoch):
        for (captions, lengths), labels, indexes in tqdm.tqdm(dataloader):

            captions = Variable(captions).to(device)
            pred = model(captions, lengths)
            
            loss = criterion(pred[1], labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print("Current Loss: ", loss.item())
        if (epoch+1) % opt.save_model == 0:
            t.save(model.state_dict(), "checkpoints/{}.pth".format(epoch))

if __name__ == "__main__":
    import fire
    fire.Fire()




    