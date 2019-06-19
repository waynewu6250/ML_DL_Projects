class Config:
    
    checkpoint = 1999
    path = './checkpoints/model.ckpt-{}'.format(checkpoint)

    num_gloabl_steps = 20000
    num_local_steps = 50

opt = Config()
