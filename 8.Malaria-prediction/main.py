import os
import torch as t
from torch.optim import Adam, SGD
from torch.autograd import Variable
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision as tv
import torch.nn as nn
import tqdm
import numpy as np

from config import opt
from data import extract_data
from resnet import ResNet, ResBlock

def train(**kwargs):

    # cuda
    device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")
    
    # attributes
    for k,v in kwargs.items():
        setattr(opt,k,v)
    
    # dataset
    train_loader = extract_data("train")
    val_loader = extract_data("val")

    # model
    # model = ResNet()
    model = tv.models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(2048, 2, bias=True)

    fc_parameters = model.fc.parameters()

    for param in fc_parameters:
        param.requires_grad = True
    
    map_location = lambda storage, loc: storage
    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path, map_location=map_location))
    model = model.to(device)

    optimizer = Adam(model.fc.parameters(), lr=opt.lr)
    criterion = t.nn.CrossEntropyLoss().to(device)

    for epoch in range(opt.max_epoch):
        train_loss = 0.0
        valid_loss = 0.0
        print("epoch %d:"%epoch)
        
        #=======Training=======#
        
        for ii, (imgs,targets) in tqdm.tqdm(enumerate(train_loader)):
            
            imgs = Variable(imgs).to(device)
            targets = Variable(targets).to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            train_loss = criterion(outputs, targets)
            train_loss.backward()
            optimizer.step()

            if ii % opt.train_loss == 0:
                print('\nTrain Epoch: {} [{}/{} ({:.0f}%)] \t Loss: {:.6f}'.format(
                epoch, ii * len(imgs), len(train_loader.dataset),
                100. * ii / len(train_loader), train_loss.item()))
        
        #=======Validation=======#
        model.eval()
        for ii, (imgs, targets) in enumerate(val_loader):
            # move to GPU
            imgs = Variable(imgs).to(device)
            targets = Variable(targets).to(device)
            ## update the average validation loss
            outputs = model(imgs)
            loss = criterion(outputs, targets)
            valid_loss = valid_loss + ((1 / (ii + 1)) * (loss.data - valid_loss))
            
        # print training/validation statistics 
        print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            epoch, 
            train_loss,
            valid_loss
            ))
        
        
        #Save model
        t.save(model.state_dict(), 'checkpoints/%s.pth' % epoch)



def test(**kwargs):

    # monitor test loss and accuracy
    correct = 0.
    total = 0.

    # cuda
    device = t.device("cuda") if t.cuda.is_available() else t.device("cpu")

    # attributes
    for k,v in kwargs.items():
        setattr(opt,k,v)

    # dataset
    test_loader = extract_data("test")

    # model
    model = tv.models.resnet50(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    model.fc = nn.Linear(2048, 2, bias=True)

    fc_parameters = model.fc.parameters()

    for param in fc_parameters:
        param.requires_grad = True
    
    map_location = lambda storage, loc: storage
    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path, map_location=map_location))
    model = model.to(device)

    optimizer = Adam(model.fc.parameters(), lr=opt.lr)
    criterion = t.nn.CrossEntropyLoss().to(device)

    for ii, (imgs,targets) in tqdm.tqdm(enumerate(test_loader)):
            
        imgs = Variable(imgs).to(device)
        targets = Variable(targets).to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        test_loss = criterion(outputs, targets)
        
        pred = outputs.data.max(1, keepdim=True)[1]
        correct += np.sum(np.squeeze(pred.eq(targets.data.view_as(pred))).cpu().numpy())
        total += imgs.size(0)
    
    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))


if __name__ == "__main__":
    import fire
    fire.Fire()