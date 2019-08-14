import torch

from torch import nn

import torch.nn.functional as F

import numpy as np

import os
import copy

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

from unet import UNet
import lossfunctions as lfs
import nibabel as nib

from dataloader_dev import RSOMLayerDataset 
from dataloader_dev import RandomZShift, ZeroCenter, CropToEven, DropBlue, ToTensor, SwapDim



def predict1(model, batch, args):
    '''
    predict one volume
    '''

    model.eval()
    
    m = batch['meta']
    print(m['dcrop']['begin'], m['dcrop']['end'], m['lcrop']['begin'], m['lcrop']['end']) 
    
    batch['data'] = batch['data'].to(
            args.device,
            args.dtype,
            non_blocking=args.non_blocking)
    
    # divide into minibatches
    minibatches = np.arange(batch['data'].shape[1],
            step=args.minibatch_size)
    # init empty prediction stack
    shp = batch['data'].shape
    print('Data shape:', shp)
    # [0 x 2 x 500 x 332]
    prediction_stack = torch.zeros((0, 2, shp[3], shp[4]),
            dtype=args.dtype,
            requires_grad = False)
    prediction_stack = prediction_stack.to(args.device)
    # print(prediction_stack.shape)

    for i2, idx in enumerate(minibatches):
        if idx + args.minibatch_size < batch['data'].shape[1]:
            data = batch['data'][:,
                    idx:idx+args.minibatch_size, :, :]
            # label = batch['label'][:,
            #         idx:idx+args.minibatch_size, :, :]
        else:
            data = batch['data'][:, idx:, :, :]
            # label = batch['label'][:,idx:, :, :]
        

        data = torch.squeeze(data, dim=0)
        prediction = model(data)
        
        # stack prediction to get volume again
        prediction = prediction.detach() 
        prediction_stack = torch.cat((prediction_stack, prediction), dim=0) 
    
    print(prediction_stack.shape)
    prediction_stack = prediction_stack.to('cpu')
    # transform -> labels
    return prediction_stack
    


def segment(pred1, pred2=None):
    ''' 
    use one or more predictions to get a segmentation
    '''
    label = (pred1[:,1,:,:] > pred1[:,0,:,:])
    print('segmentation mask shape')
    print(label.shape)

    return label
   

def predict_all(model, iterator, args, model2=None, iterator2=None, ensemble=False):
    ''' 
    predict all volumes, optional use 2 individual models for 2 views
    '''
    for i in range(args.size_pred):
        # get the next volume to evaluate 
        batch = next(iterator)
        prediction = predict1(model, batch, args)

        if ensemble:
            batch = next(iterator2)
            prediction2 = predict1(model2, batch2, args)
            label = segment(pred1=prediction, pred2=prediction2)
        else:
            label = segment(pred1 = prediction)

        label = to_numpy(label, batch['meta'])
        
        print('Label shape is:', label.shape)
        label = np.swapaxes(label, 1, 2)
        print('After swapaxes:', label.shape)
        
        filename = batch['meta']['filename'][0]
        filename = filename.replace('rgb.nii.gz','')
 
        saveNII(label, args.destination_dir, filename + 'pred')
 

def to_numpy(V, meta):
    '''
    inverse function for class ToTensor() in dataloader_dev.py 
    args
        V: torch.tensor volume
        meta: batch['meta'] information

    return V as numpy.array volume
    '''
    # torch sizes X is batch size, C is Colour
    # data
    # [X x C x Z x Y] [171 x 3 x 500-crop x 333] (without crop)
    # and for the label
    # [X x Z x Y] [171 x 500 x 333]
    
    # we want to reshape to
    # numpy sizes
    # data
    # [Z x X x Y x 3] [500 x 171 x 333 x 3]
    # label
    # [Z x X x Y] [500 x 171 x 333]
    
    # here: we only need to backtransform labels
    print(V.shape)
    if not isinstance(V, np.ndarray):
        assert isinstance(V, torch.Tensor)
        V = V.numpy()
    V = V.transpose((1, 0, 2))

    # add padding, which was removed before,
    # and saved in meta['lcrop'] and meta['dcrop']

    # structure for np.pad
    # (before0, after0), (before1, after1), ..)
    
    # parse label crop
    b = (meta['lcrop']['begin']).numpy().squeeze()
    e = (meta['lcrop']['end']).numpy().squeeze()
    print('b, e')
    print(b, e)
    print(b.shape, e.shape)
    
    pad_width = ((b[0], e[0]), (b[1], e[1]), (b[2], e[2]))
    print(V.shape)
    
    V = np.pad(V, pad_width, 'edge')

    print(V.shape)
    return V

def saveNII(V, path, fstr):
    V = V.astype(np.uint8)
    img = nib.Nifti1Image(V, np.eye(4))
    
    fstr = fstr + '.nii.gz'
    nib.save(img, os.path.join(path, fstr))


class arg_class():
    pass

args = arg_class()

os.environ["CUDA_VISIBLE_DEVICES"]='3'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

origin = '/home/gerlstefan/data/fullDataset/labeled/val'
# origin = '/home/gerlstefan/data/dataloader_dev'
destination ='/home/gerlstefan/data/prediction/test_dimswap'
model_path = '/home/gerlstefan/models/layerseg/dimswap/mod_190808_dimswap_test.pt'


# create Dataset of prediction data
dataset_pred = RSOMLayerDataset(origin,
        transform=transforms.Compose([
            ZeroCenter(),
            SwapDim(),
            CropToEven(network_depth=4),
            DropBlue(),
            ToTensor()]))

dataloader_pred = DataLoader(dataset_pred,
        batch_size=1, 
        shuffle=False, 
        num_workers=4, 
        pin_memory=True)

args.size_pred = len(dataset_pred)

print("Predicting ", args.size_pred, " Volumes.")

args.minibatch_size = 5
args.device = device
args.dtype = torch.float32
args.non_blocking = True
args.destination_dir = destination

model = UNet(in_channels=2,
             n_classes=2,
             depth=4,
             wf=6,
             padding=True,
             batch_norm=True,
             up_mode='upconv').to(args.device)
model = model.float()
model.load_state_dict(torch.load(model_path))


iterator_pred = iter(dataloader_pred)

predict_all(model, iterator_pred, args)

