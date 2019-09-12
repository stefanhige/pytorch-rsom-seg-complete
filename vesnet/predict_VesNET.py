# only predict with trained model.
import os
import torch


from VesNET import debug
from VesNET import VesNET

DEBUG = None
# DEBUG = True

pred_dir = '/home/gerlstefan/data/vesnet/synthDataset/1channel/eval'

desc = ('predict only test')
sdesc = 'predtest'

model_dir = '/home/gerlstefan/data/vesnet/out/190911-00-metrics/mod190911-00.pt'
        
os.environ["CUDA_VISIBLE_DEVICES"]='4'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


out_dir = '/home/gerlstefan/data/vesnet/out'

dirs={'train': '',
      'eval': '', 
      'model': model_dir, 
      'pred': pred_dir,
      'out': out_dir}

net1 = VesNET(device=device,
                     desc=desc,
                     sdesc=sdesc,
                     dirs=dirs,
                     divs=(3,3,3),
                     batch_size=1,
                     DEBUG=DEBUG)

net1.save_code_status()

net1.predict(use_best=False)



