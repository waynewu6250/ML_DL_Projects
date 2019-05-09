import torch as t
import torchvision as tv
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from config import opt
import tqdm

class ImageSet(Dataset):

    def __init__(self, opt):
        
        super(ImageSet, self).__init__()
        
        self.transform = tv.transforms.Compose([
            tv.transforms.Resize(224),
            tv.transforms.CenterCrop(224),
            tv.transforms.ToTensor(),
            tv.transforms.Normalize([0.485, 0.456, 0.406],
                                    [0.229, 0.224, 0.225])
        ])
        self.data = t.load(opt.caption_path)
        self.imgs = [os.path.join(opt.img_path,i) for i in self.data['id2ix'].keys()]
    
    def __getitem__(self, index):
        img = Image.open(self.imgs[index]).convert('RGB')
        img = self.transform(img)
        return img, index
    
    def __len__(self):
        return len(self.imgs)

device = t.device('cuda') if t.cuda.is_available() else t.device('cpu')

#Data
data = ImageSet(opt)
dataloader = DataLoader(data, batch_size=opt.batch_size, shuffle=False)
results = t.zeros([len(data), 2048])

#Model
model = tv.models.resnet50(pretrained=True)
del model.fc
model.fc = lambda x: x
model.to(device)

for ii, (imgs,indexes) in tqdm.tqdm(enumerate(dataloader)):

    imgs = imgs.to(device)
    outputs = model(imgs)
    results[ii*opt.batch_size:(ii+1)*opt.batch_size] = outputs.data.cpu()

t.save(results, 'image_feature.pth')






        

    
