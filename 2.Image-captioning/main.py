import torch as t
import torchvision as tv
from torch.optim import Adam
from torch.nn.utils.rnn import pack_padded_sequence
from torch.autograd import Variable
import os
import tqdm
from PIL import Image

from model import CaptionModel
from all_data import get_dataloader
from config import opt

def train(**kwargs):

    device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')
    for k,v in kwargs.items():
        setattr(opt, k, v)
    
    dataloader = get_dataloader(opt)
    model = CaptionModel(opt, dataloader.dataset.word2ix, dataloader.dataset.id2ix)
    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path))
    t.backends.cudnn.enabled = False
    model = model.to(device)
    
    optimizer = Adam(model.parameters(), opt.lr)
    criterion = t.nn.CrossEntropyLoss()
    for epoch in range(opt.max_epoch):
        for ii, (imgs, (captions, lengths), indexes) in tqdm.tqdm(enumerate(dataloader)):

            imgs = Variable(imgs).to(device)
            captions = Variable(captions).to(device)
            pred, _ = model(imgs, captions, lengths)
            target_captions = pack_padded_sequence(captions, lengths)[0]
            
            loss = criterion(pred, target_captions)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print("Current Loss: ", loss.item())
        if (epoch+1) % opt.save_model == 0:
            t.save(model.state_dict(), "checkpoints/{}.pth".format(epoch))

def generate(**kwargs):

    device=t.device('cuda') if t.cuda.is_available else t.device('cpu')
    for k, v in kwargs.items():
        setattr(opt, k, v)

    data = t.load(opt.caption_path, map_location=lambda s, l: s)
    word2ix, ix2word = data['word2ix'], data['ix2word']

    transforms = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ])
    img = Image.open(opt.test_img)
    img = transforms(img).unsqueeze(0)

    resnet50 = tv.models.resnet50(True).eval()
    del resnet50.fc
    resnet50.fc = lambda x: x
    # resnet50 = resnet50.to(device)
    # img = img.to(device)
    img_feats = resnet50(img).detach()

    # Caption Model
    model = CaptionModel(opt, word2ix, ix2word)
    model.load_state_dict(t.load(opt.model_path))
    # model.to(device)

    results = model.generate(img_feats.data[0])
    print('\r\n'.join(results))

if __name__ == "__main__":
    import fire
    fire.Fire()




    


    


