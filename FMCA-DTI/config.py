# -*- coding:utf-8 -*-

class hyperparameter():
    def __init__(self):
        self.Learning_rate = 1e-5 
        self.Epoch = 100 
        self.Batch_size = 64 
        self.Patience = 50
        self.decay_interval = 10
        self.lr_decay = 0.5
        self.weight_decay = 1e-4
        self.protein_kernel = [4,6,8]
        self.drug_kernel = [3,3,3]
        self.conv = 128  
        self.char_dim = 512  
        self.loss_epsilon = 1

def FMCAargs():
    config = {
             'embed_d_size': 512,
              'embed_p_size': 512,
              'd_channel_size': [[19,512],[19,256, 512],[19,128,256, 512]],
              'p_channel_size': [[181,512],[19,256, 512],[19,128,256, 512]],
              'filter_d_size': [32,32, 32,32],
              'filter_p_size': [32,32, 64],
              'num_embedding': 32,
              'fc_size': [1024, 512, 256,],
              'clip':True,
              }
    config['max_drug_seq'] = {"celegans": [19, 11], "human": [20, 21], "BIOSNAP": [19, 17], "DAVIS": [8, 11]}
    config['max_protein_seq'] = {"celegans": 181, "human": 184, "BIOSNAP": 184, "DAVIS": 156}

    config['input_d_dim']={"celegans":[2184,1804],"human":[3269,2658],"BIOSNAP":[5733,4269],"DAVIS":[184,168]}
    config['input_p_dim']={"celegans":224,"human":226,"BIOSNAP":229,"DAVIS":225}
    return config
