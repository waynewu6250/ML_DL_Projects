import sys, os
import torch as t
from torch.autograd import Variable
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import DataLoader

from data.data import get_data
from model import LyricsModel
from config import opt
from visualize import Visualizer
from torchnet import meter

import tqdm

def train(**kwargs):
    
    #Set attributes
    for k,v in kwargs.items():
        setattr(opt, k, v)
    if opt.vis:
        visualizer = Visualizer()
    
    # Data
    data, word2ix, ix2word = get_data(opt)
    data = t.from_numpy(data)
    dataloader = DataLoader(data, batch_size=opt.batch_size, shuffle=True)

    # Model
    model = LyricsModel(len(word2ix), opt.embedding_dim, opt.latent_dim)
    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path))
    
    # Define optimizer and loss
    optimizer = Adam(model.parameters(), lr=opt.lr)
    criterion = nn.CrossEntropyLoss()
    loss_meter = meter.AverageValueMeter()

    if opt.use_gpu:
        model.cuda()
        criterion.cuda()

    #================================================#
    #               Start Training                   #
    #================================================#
    
    for epoch in tqdm.tqdm(range(opt.num_epoch)):
        
        for (ii, data) in enumerate(dataloader):
            # Prepare data
            data = data.long().transpose(1,0).contiguous()
            if opt.use_gpu: data = data.cuda()
            inputs, targets = Variable(data[:-1,:]), Variable(data[1:,:])
            outputs,hidden = model(inputs)

            # Initialize and backward
            optimizer.zero_grad()
            loss = criterion(outputs,targets.view(-1))
            loss.backward()
            optimizer.step()

            loss_meter.add(loss.item())

            if (1+ii) % opt.print_every == 0:
                print("Current Loss: %d"%loss.item())
                if opt.vis:
                    visualizer.plot('loss', loss_meter.value()[0])
        if (epoch+1) % 20 == 0:
            t.save(model.state_dict(), 'checkpoints/%s.pth'%epoch)


def generate(**kwargs):

    #Set attributes
    for k,v in kwargs.items():
        setattr(opt, k, v)
    
    # Data
    data, word2ix, ix2word = get_data(opt)

    # Load model
    model = LyricsModel(len(word2ix), opt.embedding_dim, opt.latent_dim)
    if opt.model_path:
        model.load_state_dict(t.load(opt.model_path,map_location=lambda s,l: s))
        
    #================================================#
    #               Start Decoding                   #
    #================================================#

    results = list(opt.start_words)
    input_ = Variable(t.LongTensor([word2ix["<START>"]])).view(1,1)
    hidden = None

    if opt.use_gpu:
        model.cuda()
        input_ = input_.cuda()
    
    if opt.prefix_words:
        for w in opt.prefix_words:
            output,hidden = model(input_,hidden)
            input_ = Variable(t.LongTensor([word2ix[w]])).view(1,1)
    
    for i in range(opt.max_gen_len):

        output,hidden = model(input_,hidden)
        
        if i < len(opt.start_words):
            word = opt.start_words[i]
            input_ = Variable(t.LongTensor([word2ix[word]])).view(1,1)
        else:
            top_index = output.data[0].topk(1)[1]
            word = ix2word[top_index.data.item()]
            results.append(word)
            input_ = Variable(t.LongTensor([top_index])).view(1,1)
        if word == '<EOS>':
            break
    
    if "<EOS>" in results:
        results.remove("<EOS>")
    print(''.join(results).rstrip())


if __name__ == '__main__':
    import fire
    fire.Fire()
    
    
    
