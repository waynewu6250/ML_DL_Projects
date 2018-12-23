import visdom
import time
import numpy as np

class Visualizer():
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}
        self.log_text = ''
    

    #Change the setting
    def init(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self
    
    #Plot
    def plot(self, name, loss, **kwargs):
        x = self.index.get(name)
        self.vis.line(X=np.array([x]), Y=np.array([loss]), 
                      win=(name), 
                      opts=dict(title=name),
                      update = None if x == 0 else 'append',
                      **kwargs)
        self.index[name] = x+1
    
    def plot_many(self, d):
        for k,v in d.items():
            self.plot(k,v)
    

    #Image
    def img(self, name, _img, **kwargs):
        self.vis.images(_img,
                        win=(name),
                        opts=dict(title=name),
                        **kwargs)
    
    def img_many(self, d):
        for k,v in d.items():
            self.img(k,v)
    
    #Log_text
    def log(self, info):
        self.log_text = '{time} {info} <br>'.format(time=time.strftime('%m%d_%H%M%S'), info=info)
        self.vis.text(self.log_text, win='log_text1')
    
    #Get function
    def __getattr__(self, name):
        return getattr(self.vis, name)