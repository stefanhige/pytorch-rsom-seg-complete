# class for one CNN experiment

import os
import sys
import copy
import json
import warnings
from timeit import default_timer as timer

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import numpy as np
from scipy import ndimage

from ._model import UNet
import laynet._metrics as lfs #lfs=lossfunctions
from ._dataset import RSOMLayerDataset, RSOMLayerDatasetUnlabeled, \
                      RandomZShift, ZeroCenter, CropToEven, DropBlue, \
                      ToTensor, precalcLossWeight, SwapDim, to_numpy
from utils import save_nii

class LayerNetBase():
    """
    stripped base class for predicting RSOM layers.
    for training user class LayerNet
    Args:
        device             torch.device()     'cuda' 'cpu'
        dirs               dict of string      use these directories
        filename           string              pattern to save output
    """
    def __init__(self, 
                 dirs={'train':'', 'eval':'', 'pred':'', 'model':'', 'out':''},
                 device=torch.device('cuda'),
                 model_depth=4
                 ):

        self.model_depth = model_depth
        self.dirs = dirs

        self.pred_dataset = RSOMLayerDatasetUnlabeled(
                dirs['pred'],
                transform=transforms.Compose([
                    ZeroCenter(), 
                    CropToEven(network_depth=self.model_depth),
                    DropBlue(),
                    ToTensor()])
                )

        self.pred_dataloader = DataLoader(
            self.pred_dataset,
            batch_size=1, 
            shuffle=False, 
            num_workers=4, 
            pin_memory=True)


        self.size_pred = len(self.pred_dataset)

        self.minibatch_size = 1
        self.device = device
        self.dtype = torch.float32
        
        self.model = UNet(in_channels=2,
                          n_classes=2,
                          depth=self.model_depth,
                          wf=6,
                          padding=True,
                          batch_norm=True,
                          up_mode='upconv').to(self.device)
        
        if self.dirs['model']:
            self.model.load_state_dict(torch.load(self.dirs['model']))
        

    def predict(self):
        self.model.eval()
        iterator = iter(self.pred_dataloader) 

        for i in range(self.size_pred):
            # get the next volume to evaluate 
            batch = next(iterator)
            
            m = batch['meta']
            
            batch['data'] = batch['data'].to(
                    self.device,
                    self.dtype,
                    non_blocking=True
                    )
            
            # divide into minibatches
            minibatches = np.arange(batch['data'].shape[1],
                                    step=self.minibatch_size)
            # init empty prediction stack
            shp = batch['data'].shape
            # [0 x 2 x 500 x 332]
            prediction_stack = torch.zeros((0, 2, shp[3], shp[4]),
                    dtype=self.dtype,
                    requires_grad=False
                    )
            prediction_stack = prediction_stack.to(self.device)

            for i2, idx in enumerate(minibatches):
                if idx + self.minibatch_size < batch['data'].shape[1]:
                    data = batch['data'][:, idx:idx+self.minibatch_size, :, :]
                else:
                    data = batch['data'][:, idx:, :, :]
     
                data = torch.squeeze(data, dim=0)

                prediction = self.model(data)

                prediction = prediction.detach() 
                prediction_stack = torch.cat((prediction_stack, prediction), dim=0) 
            
            prediction_stack = prediction_stack.to('cpu')
            
            
            # transform -> labels
            label = (prediction_stack[:,1,:,:] > prediction_stack[:,0,:,:]) 

            m = batch['meta']

            label = to_numpy(label, m)

            filename = batch['meta']['filename'][0]
            filename = filename.replace('rgb.nii.gz','')
            label = self.smooth_pred(label, filename)

            print('Saving', filename)
            save_nii(label, self.dirs['out'], filename + 'pred')
            

            # compare to ground truth
            if 0:
                label_gt = batch['label']
          
                label_gt = torch.squeeze(label_gt, dim=0)
                label_gt = to_numpy(label_gt, m)

                label_diff = (label > label_gt).astype(np.uint8)
                label_diff += 2*(label < label_gt).astype(np.uint8)
                # label_diff = label != label_gt
                save_nii(label_diff, self.dirs['out'], filename + 'dpred')

    @staticmethod
    def smooth_pred(label, filename):
        '''
        smooth the prediction
        '''
        
        # 1. fill holes inside the label
        ldtype = label.dtype
        label = ndimage.binary_fill_holes(label).astype(ldtype)
        label_shape = label.shape
        label = np.pad(label, 2, mode='edge')
        label = ndimage.binary_closing(label, iterations=2)
        label = label[2:-2,2:-2,2:-2]
        assert label_shape == label.shape
        
        # 2. scan along z-dimension change in label 0->1 1->0
        #    if there's more than one transition each, one needs to be dropped
        #    after filling holes, we hope to be able to drop the outer one
        # 3. get 2x 2-D surface data with surface height being the index in z-direction
        
        surf_lo = np.zeros((label_shape[1], label_shape[2]))
        
        # set highest value possible (500) as default. Therefore, empty sections
        # of surf_up and surf_lo will get smoothened towards each other, and during
        # reconstructions, we won't have any weird shapes.
        surf_up = surf_lo.copy()+label_shape[0]

        for xx in np.arange(label_shape[1]):
            for yy in np.arange(label_shape[2]):
                nz = np.nonzero(label[:,xx,yy])
                
                if nz[0].size != 0:
                    idx_up = nz[0][0]
                    idx_lo = nz[0][-1]
                    surf_up[xx,yy] = idx_up
                    surf_lo[xx,yy] = idx_lo
       
        #    smooth coarse structure, eg with a 25x25 average and crop everything which is above average*factor
        #           -> hopefully spikes will be removed.
        surf_up_m = ndimage.median_filter(surf_up, size=(26, 26), mode='nearest')
        surf_lo_m = ndimage.median_filter(surf_lo, size=(26, 26), mode='nearest')
        
        for xx in np.arange(label_shape[1]):
            for yy in np.arange(label_shape[2]):
                if surf_up[xx,yy] < surf_up_m[xx,yy]:
                    surf_up[xx,yy] = surf_up_m[xx,yy]
                if surf_lo[xx,yy] > surf_lo_m[xx,yy]:
                    surf_lo[xx,yy] = surf_lo_m[xx,yy]

        # apply suitable kernel in order to smooth
        # smooth fine structure, eg with a 5x5 moving average
        surf_up = ndimage.uniform_filter(surf_up, size=(9, 5), mode='nearest')
        surf_lo = ndimage.uniform_filter(surf_lo, size=(9, 5), mode='nearest')

        # 5. reconstruct label
        label_rec = np.zeros(label_shape, dtype=np.uint8)
        for xx in np.arange(label_shape[1]):
            for yy in np.arange(label_shape[2]):

                label_rec[int(np.round(surf_up[xx,yy])):int(np.round(surf_lo[xx,yy])),xx,yy] = 1     

        return label_rec


class LayerUNET():
    '''
    class for setting up, training and evaluating of layer segmentation
    with unet on RSOM dataset
    Args:
        device             torch.device()     'cuda' 'cpu'
        model_depth        int                 unet depth
        dataset_zshift     int or (int, int)   data aug. zshift
        dirs               dict of string      use these directories
        filename           string              pattern to save output
        optimizer          string
        initial_lr         float               initial learning rate
        scheduler_patience int                 n epochs before lr reduction
        lossfn             function            custom lossfunction
        class_weight       (float, float)      class weight for classes (0, 1)
        epochs             int                 number of epochs 
    '''
    def __init__(self,
                 device=torch.device('cuda'),
                 model_depth=3,
                 dataset_zshift=0,
                 dirs={'train':'','eval':'', 'model':'', 'pred':''},
                 filename = '',
                 optimizer = 'Adam',
                 initial_lr = 1e-4,
                 scheduler_patience = 3,
                 lossfn = lfs.custom_loss_1,
                 lossfn_smoothness = 0,
                 class_weight = None,
                 epochs = 30,
                 dropout = False
                 ):
        
        # PROCESS LOGGING
        self.filename = filename
        try:
            self.logfile = open(os.path.join(dirs['model'], 'log_' + filename), 'x')
        except:
            self.logfile = open(os.path.join(dirs['model'], 'log_' + filename), 'a')
            warnings.warn('logfile already exists! appending to existing file..', UserWarning) 
        
        # MODEL
        self.model = UNet(in_channels=2,
             n_classes=2,
             depth=model_depth,
             wf=6,
             padding=True,
             batch_norm=True,
             up_mode='upconv',
             dropout=dropout)
        self.model_dropout = dropout
        
        self.model = self.model.to(device)
        self.model = self.model.float()
        
        print(self.model.down_path[0].block.state_dict()['0.weight'].device)

        self.model_depth = model_depth
        
        # LOSSFUNCTION
        self.lossfn = lossfn
        if class_weight is not None:
            self.class_weight = torch.tensor(class_weight, dtype=torch.float32)
            self.class_weight = self.class_weight.to(device)
        else:
            self.class_weight = None

        self.lossfn_smoothness = lossfn_smoothness
        
        
        # DIRECTORIES
        # Dictionary with entries 'train' 'eval' 'model' 'pred'
        self.dirs = dirs

        
        # DATASET
        self.train_dataset_zshift = dataset_zshift
        
        self.train_dataset = RSOMLayerDataset(self.dirs['train'],
            transform=transforms.Compose([RandomZShift(dataset_zshift),
                                          ZeroCenter(),
                                          # SwapDim(),
                                          CropToEven(network_depth=self.model_depth),
                                          DropBlue(),
                                          ToTensor(),
                                          precalcLossWeight()]))
        
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_size=1, 
                                           shuffle=True, 
                                           num_workers=4, 
                                           pin_memory=True)



        
        self.eval_dataset = RSOMLayerDataset(self.dirs['eval'],
            transform=transforms.Compose([RandomZShift(),
                                          ZeroCenter(),
                                          # SwapDim(),
                                          CropToEven(network_depth=self.model_depth),
                                          DropBlue(),
                                          ToTensor(),
                                          precalcLossWeight()]))
        self.eval_dataloader = DataLoader(self.eval_dataset,
                                          batch_size=1, 
                                          shuffle=False, 
                                          num_workers=4, 
                                          pin_memory=True)
        
        
        # OPTIMIZER
        self.initial_lr = initial_lr
        if optimizer == 'Adam':
            self.optimizer = torch.optim.Adam(
                    self.model.parameters(),
                    lr=self.initial_lr,
                    weight_decay = 0
                    )
        
        # SCHEDULER
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, 
                mode='min', 
                factor=0.1,
                patience=scheduler_patience,
                verbose=True,
                threshold=1e-4,
                threshold_mode='rel',
                cooldown=0,
                min_lr=0,
                eps=1e-8)
        
        # HISTORY
        self.history = {
                'train':{'epoch': [], 'loss': []},
                'eval':{'epoch': [], 'loss': []}
                }
        
        # CURRENT EPOCH
        self.curr_epoch = None
        
        # ADDITIONAL ARGS
        self.args = self.helperClass()
        
        self.args.size_train = len(self.train_dataset)
        self.args.size_eval = len(self.eval_dataset)
        self.args.minibatch_size = 5
        self.args.device = device
        self.args.dtype = torch.float32
        self.args.non_blocking = True
        self.args.n_epochs = epochs
        self.args.data_dim = self.eval_dataset[0]['data'].shape
        
    def printConfiguration(self, destination='stdout'):
        if destination == 'stdout':
            where = sys.stdout
        elif destination == 'logfile':
            where = self.logfile
        
        print('LayerUNET configuration:',file=where)
        print('DATA: train dataset loc:', self.dirs['train'], file=where)
        print('      train dataset len:', self.args.size_train, file=where)
        print('      eval dataset loc:', self.dirs['eval'], file=where)
        print('      eval dataset len:', self.args.size_eval, file=where)
        print('      shape:', self.args.data_dim, file=where)
        print('      zshift:', self.train_dataset_zshift)
        print('EPOCHS:', self.args.n_epochs, file=where)
        print('OPTIMIZER:', self.optimizer, file=where)
        print('initial lr:', self.initial_lr, file=where)
        print('LOSS: fn', self.lossfn, file=where)
        print('      class_weight', self.class_weight, file=where)
        print('      smoothnes param', self.lossfn_smoothness, file=where)
        print('CNN:  unet', file=where)
        print('      depth', self.model_depth, file=where)
        print('      dropout?', self.model_dropout, file=where)
        print('OUT:  model:', self.dirs['model'], file=where)
        print('      pred:', self.dirs['pred'], file=where)

    def train_all_epochs(self):  
        self.best_model = copy.deepcopy(self.model.state_dict())
        for k, v in self.best_model.items():
            self.best_model[k] = v.to('cpu')
        
        self.best_loss = float('inf')
        
        print('Entering training loop..')
        for curr_epoch in range(self.args.n_epochs): 
            # in every epoch, generate iterators
            train_iterator = iter(self.train_dataloader)
            eval_iterator = iter(self.eval_dataloader)
            
            curr_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
        
            if curr_epoch == 1:
                tic = timer()
            
            self.train(iterator=train_iterator, epoch=curr_epoch)
            
            torch.cuda.empty_cache()
            if curr_epoch == 1:
                toc = timer()
                print('Training took:', toc - tic)
                tic = timer()
            
            self.eval(iterator=eval_iterator, epoch=curr_epoch)
        
            torch.cuda.empty_cache()
            if curr_epoch == 1:
                toc = timer()
                print('Evaluation took:', toc - tic)
                
            print(torch.cuda.memory_cached()*1e-6,'MB memory used')
            # extract the average training loss of the epoch
            le_idx = self.history['train']['epoch'].index(curr_epoch)
            le_losses = self.history['train']['loss'][le_idx:]
            # divide by batch size (170) times dataset size
            train_loss = sum(le_losses) / (self.args.data_dim[0]*self.args.size_train)
            
            # extract most recent eval loss
            curr_loss = self.history['eval']['loss'][-1]
            
            # use ReduceLROnPlateau scheduler
            self.scheduler.step(curr_loss)
            
            if curr_loss < self.best_loss:
                self.best_loss = copy.deepcopy(curr_loss)
                self.best_model = copy.deepcopy(self.model.state_dict())
                for k, v in self.best_model.items():
                    self.best_model[k] = v.to('cpu')
                found_nb = 'new best!'
            else:
                found_nb = ''
        
            print('Epoch {:d} of {:d}: lr={:.0e}, Lt={:.2e}, Le={:.2e}'.format(
                curr_epoch+1, self.args.n_epochs, curr_lr, train_loss, curr_loss), found_nb)
            print('Epoch {:d} of {:d}: lr={:.0e}, Lt={:.2e}, Le={:.2e}'.format(
                curr_epoch+1, self.args.n_epochs, curr_lr, train_loss, curr_loss), found_nb, file=self.logfile)
    
        print('finished. saving model')
        self.logfile.close()
    
    def train(self, iterator, epoch):
        '''
        train one epoch
        Args:   model
                iterator
                optimizer
                history
                epoch
                lossfn
                args 
        '''
        # PARSE
        # model = self.model
        # optimizer = self.optimizer
        # history = self.history
        # lossfn = self.lossfn
        # args = self.args
        
        self.model.train()
        
        for i in range(self.args.size_train):
            # get the next batch of training data
            batch = next(iterator)
            
            # label_ = batch['label']
            # print(label_.shape)
            
            # print(label_[:,:,0,:].sum().item())
            # print(label_[:,:,-1,:].sum().item())
                    
            batch['label'] = batch['label'].to(
                    self.args.device, 
                    dtype=self.args.dtype, 
                    non_blocking=self.args.non_blocking)
            batch['data'] = batch['data'].to(
                    self.args.device,
                    self.args.dtype,
                    non_blocking=self.args.non_blocking)
            batch['meta']['weight'] = batch['meta']['weight'].to(
                    self.args.device,
                    self.args.dtype,
                    non_blocking=self.args.non_blocking)
        
        
            # divide into minibatches
            minibatches = np.arange(batch['data'].shape[1],
                    step=self.args.minibatch_size)
            for i2, idx in enumerate(minibatches): 
                if idx + self.args.minibatch_size < batch['data'].shape[1]:
                    data = batch['data'][:,
                            idx:idx+self.args.minibatch_size, :, :]
                    label = batch['label'][:,
                            idx:idx+self.args.minibatch_size, :, :]
                    weight = batch['meta']['weight'][:,
                            idx:idx+self.args.minibatch_size, :, :]
                else:
                    data = batch['data'][:, idx:, :, :]
                    label = batch['label'][:, idx:, :, :]
                    weight = batch['meta']['weight'][:, idx:, :, :]
                
         
                data = torch.squeeze(data, dim=0)
                label = torch.squeeze(label, dim=0)
                weight = torch.squeeze(weight, dim=0)
                
                prediction = self.model(data)
            
                # move back to save memory
                # prediction = prediction.to('cpu')
                loss = self.lossfn(
                        pred=prediction, 
                        target=label,
                        spatial_weight=weight,
                        class_weight=self.class_weight,
                        smoothness_weight = self.lossfn_smoothness)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        
                frac_epoch = epoch +\
                        i/self.args.size_train +\
                        i2/(self.args.size_train * minibatches.size)
                
                # print(epoch, i/args.size_train, i2/minibatches.size)
                self.history['train']['epoch'].append(frac_epoch)
                self.history['train']['loss'].append(loss.data.item())
                
    def eval(self, iterator, epoch):
        '''
        evaluate with the validation set
        Args:   model
                iterator
                optimizer
                history
                epoch
                lossfn
                args
        '''
        # PARSE
        # model = self.model
        # history = self.history
        # lossfn = self.lossfn
        # args = self.args
        
        
        self.model.eval()
        running_loss = 0.0
        
        for i in range(self.args.size_eval):
            # get the next batch of the testset
            
            batch = next(iterator)
            batch['label'] = batch['label'].to(
                    self.args.device, 
                    dtype=self.args.dtype, 
                    non_blocking=self.args.non_blocking)
            batch['data'] = batch['data'].to(
                    self.args.device,
                    self.args.dtype,
                    non_blocking=self.args.non_blocking)
            batch['meta']['weight'] = batch['meta']['weight'].to(
                    self.args.device,
                    self.args.dtype,
                    non_blocking=self.args.non_blocking)
        
            # divide into minibatches
            minibatches = np.arange(batch['data'].shape[1],
                    step=self.args.minibatch_size)
            for i2, idx in enumerate(minibatches):
                if idx + self.args.minibatch_size < batch['data'].shape[1]:
                    data = batch['data'][:,
                            idx:idx+self.args.minibatch_size, :, :]
                    label = batch['label'][:,
                            idx:idx+self.args.minibatch_size, :, :]
                    weight = batch['meta']['weight'][:,
                            idx:idx+self.args.minibatch_size, :, :]
                else:
                    data = batch['data'][:, idx:, :, :]
                    label = batch['label'][:,idx:, :, :]
                    weight = batch['meta']['weight'][:, idx:, :, :]
         
                data = torch.squeeze(data, dim=0)
                label = torch.squeeze(label, dim=0)
                weight = torch.squeeze(weight, dim=0)
                
                prediction = self.model(data)
        
                # prediction = prediction.to('cpu')
                loss = self.lossfn(
                        pred=prediction, 
                        target=label,
                        spatial_weight=weight,
                        class_weight=self.class_weight,
                        smoothness_weight=self.lossfn_smoothness)
                
                # loss running variable
                # TODO: check if this works
                # add value for every minibatch
                # this should scale linearly with minibatch size
                # have to verify!
                running_loss += loss.data.item()
                
                # adds up all the dice coeeficients of all samples
                # processes each slice individually
                # in the end need to divide by number of samples*number of slices per sample
                # in the end it needs to divided by the number of iterations
                # running_dice += self.dice_coeff(pred=prediction,
                #                     target=label)
        
            # running_loss adds up loss for every batch and minibatch,
            # divide by size of testset*size of each batch
            epoch_loss = running_loss / (self.args.size_eval*batch['data'].shape[1])
            self.history['eval']['epoch'].append(epoch)
            self.history['eval']['loss'].append(epoch_loss)
           
    def calc_weight_std(self, model):
        '''
        calculate the standard deviation of all weights in model_dir

        '''
        if isinstance(model, torch.nn.Module):
            model = model.state_dict()
        
        all_values = np.array([])

        for name, values in model.items():
            if 'weight' in name:
                values = values.to('cpu').numpy()
                values = values.ravel()
                all_values = np.concatenate((all_values, values))

        stdd = np.std(all_values)
        mean = np.mean(all_values)
        print('model number of weights:', len(all_values))
        print('model weights standard deviation:', stdd)
        print('model weights mean value:        ', mean)

    def jaccard_index(pred, target):
        '''
        calculate the jaccard index per slice and return
        the sum of jaccard indices
        '''
        # TODO: implementation never used or tested

        # shapes
        # [slices, x, x]

        pred_shape = pred.shape
        print(pred.shape)

        # for every slice
        jaccard_sum = 0.0
        for slc in range(pred_shape[0]):
            pflat = pred[slc, :, :]
            tflat = target[slc, :, :]
            intersection = (pflat * tflat).sum()
            jaccard_sum += intersection/(pflat.sum() + tflat.sum())
            
        return jaccard_sum

    def save(self):
        torch.save(self.best_model, os.path.join(self.dirs['model'], 'mod_' + self.filename + '.pt'))
        
        json_f = json.dumps(self.history)
        f = open(os.path.join(self.dirs['model'],'hist_' + self.filename + '.json'),'w')
        f.write(json_f)
        f.close()
            
    class helperClass():
        pass
        
        
# EXECUTION TEST
# train_dir = '/home/gerlstefan/data/dataloader_dev'
# eval_dir = train_dir

# try 4 class weights
if __name__ == '__main__':
    N = 1


    root_dir = '/home/gerlstefan/data/fullDataset/labeled'


    model_name = '190808_dimswap_test'


    model_dir = '/home/gerlstefan/models/layerseg/dimswap'
            
    os.environ["CUDA_VISIBLE_DEVICES"]='6'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for idx in range(N):
        root_dir = root_dir

        print('current model')
        print(model_name, root_dir)
        train_dir = os.path.join(root_dir, 'train')
        eval_dir = os.path.join(root_dir, 'val')
        dirs={'train':train_dir,'eval':eval_dir, 'model':model_dir, 'pred':''}

        net1 = LayerUNET(device=device,
                             model_depth=4,
                             dataset_zshift=(-50, 200),
                             dirs=dirs,
                             filename=model_name,
                             optimizer='Adam',
                             initial_lr=1e-4,
                             scheduler_patience=3,
                             lossfn=lfs.custom_loss_1_smooth,
                             lossfn_smoothness = 50,
                             epochs=30,
                             dropout=True,
                             class_weight=(0.3, 0.7),
                             )

        net1.printConfiguration()
        net1.printConfiguration('logfile')
        print(net1.model, file=net1.logfile)

        net1.train_all_epochs()
        net1.save()

