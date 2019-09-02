import torch
import torch.nn.functional as F
import torch.nn as nn
import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
# this whole file is copied from
# https://github.com/ramialmask/pytorch_deepvessel/tree/master/components
# thanks to Rami Al-Maskari



#Define Network
class Deep_Vessel_Net_FC(nn.Module):

    def __init__(self, 
                 in_channels=2, 
                 print_=False): #whats with n_classes? 
        super(Deep_Vessel_Net_FC, self).__init__()
        self.in_channels = in_channels

        # ONLY FOR TESTING PURPOSES
        self.print=print_

        # 1. 3x3x3-Conv, 2 -> 5
        self.conv1 = nn.Conv3d(self.in_channels, 5, kernel_size=(3,3,3)) #nn.Conv3d
        # 2. 5x5x5-Conv, 5 -> 10
        self.conv2 = nn.Conv3d(5, 10, kernel_size=(5,5,5))
        # 3. 5x5x5-Conv, 10-> 20
        self.conv3 = nn.Conv3d(10, 20, kernel_size=(5,5,5))
        # 4. 3x3x3-Conv, 20-> 50
        self.conv4 = nn.Conv3d(20, 50, kernel_size=(3,3,3))
        # 5. FC
        self.conv5 = nn.Conv3d(50,1, kernel_size=(1,1,1))#, bias=False)

    def set_padding(self, size):
        """ONLY FOR TESTING PURPOSES
        """
        self.padding = nn.ReplicationPad3d(size)
        
    def calc_shrink(self):
        X = torch.ones([1, self.in_channels, 25, 25, 25])
        Y = self.forward(X)
        Xshp = torch.tensor(list(X.shape))
        Yshp = torch.tensor(list(Y.shape))
        print(' ',list(Xshp.numpy()),'input dimension')
        print('-',list(Yshp.numpy()),'output dimension')
        print('______________________')
        print('=',list((Xshp-Yshp).numpy()))

    def print_volume(self, x, layer):
        """ONLY FOR TESTING PURPOSES
        set self.c = 0 in init
        set self.c = self.c + 1 in forward
        """
        if self.c == 30 and self.print:
            image = x.data.clone()
            image = image.detach().cpu().numpy()
            print("Image shape before {}".format(image.shape))
            image = image[0,0,:,:,0]
            print("Image shape after {}".format(image.shape))
            image = Image.fromarray(image)
            img_path = "segmentation/output/synth/wave/" + "image_{}_{}.tiff".format(self.c, layer)
            image.save(img_path)
            print("Saved image to {}".format(img_path))

    def forward(self, x):
        
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = self.conv5(x)
        return x

    def weight_hist(self, gs):
        for i, c in enumerate(self.modules()):
            #print("{} {}".format(i,c._get_name()))
            if i == 0:
                pass
            else:
                # print("plotting {}".format(i))
                w = c.weight.clone().detach().cpu().numpy()
                w = w.ravel()
                me = np.mean(w)
                std = np.std(w)
                mi = np.amin(w)
                ma = np.amax(w)
                # print("{}, {}".format(str(mi),str(ma)))
                le = len(w)
                plt.subplot(gs[i-1,0])
                plt.hist(w, bins=100,facecolor="g",label="mean = {:.4f}\n \
                        std = {:.4f}\n \
                        [{:.4f},{:.4f}]\n \
                        len = {}\n".format(me, std, mi, ma, le))
                plt.legend(loc="best", handletextpad=-2.0, handlelength=0)
                plt.title("{} {} Weight".format(c._get_name(), str(i)))
                if not i == 6:
                    b = c.bias.clone().detach().cpu().numpy()
                    b = b.ravel()
                    me = np.mean(b)
                    std = np.std(b)
                    mi = np.amin(b)
                    ma = np.amax(b)
                    le = len(b)
                    plt.subplot(gs[i-1,1])
                    plt.hist(b, bins=100,facecolor="g",label="mean = {:.4f}\n \
                        std = {:.4f}\n \
                        [{:.4f},{:.4f}]\n \
                        len = {}\n".format(me, std, mi, ma, le))
                    plt.legend(loc="best", handletextpad=-2.0, handlelength=0)
                    plt.title("{} {} Bias".format(c._get_name(), str(i)))

    def print_weights(self, settings):
        """ONLY FOR TESTING PURPOSES
        """
        from utilities.filehandling import writeNifti
        import os
        path="./segmentation/output/test_dir/networks/" + settings["paths"]["input_model"][:-4] + "/"
        os.mkdir(path)
        print("\t\tWeights\t\t\t\tBias\t\tnelements\tsize")
        for i, c in enumerate(self.modules()):
            try:
                writeNifti(path + "layer_{}.nii".format(i), c.weight.clone().detach().cpu().numpy())
                if i == 1:
                    print("{}\t:{}\t{}\t\t{}\t\t{}".format(str(i), \
                            c.weight.shape,
                            c.bias.shape,
                            c.weight.nelement() + c.bias.nelement(),
                            c.weight.nelement() * c.weight.element_size() + \
                            c.bias.nelement() * c.bias.element_size()))
                # if i == 2:
                #     print("First layer weight {}".format(c.weight[-1]))
                if i > 1:
                    print("{}\t:{}\t{}\t{}\t\t{}".format(str(i), \
                            c.weight.shape,
                            c.bias.shape,
                            c.weight.nelement() + c.bias.nelement(),
                            c.weight.nelement() * c.weight.element_size() + \
                            c.bias.nelement() * c.bias.element_size()))
            except AttributeError:
                pass

    def print_layer(self, num):
        module = None
        for i, c in enumerate(self.modules()):
            if i == num:
                module = c
                break
        print("Layer {} Weight {}".format(num, module.weight[-1]))

    def get_module_length(self):
        return len(list(self.modules()))

    def get_layer(self, num):
        module = None
        for i, c in enumerate(self.modules()):
            if i == num:
                module = c
                break
        return module

    def save_model(self, path):
        torch.save(self.state_dict(), path)

    def load_model(self, path):
        t_ = torch.load(path)
        print("T_ {}".format(t_))
        self.load_state_dict(t_)
        print("Loaded model from {0}".format(str(path)))

class DeepVesselNet(nn.Module):

    def __init__(self, in_channels=2, depth = 5, dropout=False, batchnorm=False):
        super(DeepVesselNet, self).__init__()
       
        print('Calling DeepVesselNet init')
        # SETTINGS
        self.in_channels = in_channels
        self.depth = 4
        self.dropout = dropout
       
        # generate dropout list for every layer
        if dropout:
            self.dropouts = [0, 0.3, 0.3, 0.3, 0]
        else:
            self.dropouts = [0] * depth

        # generate channels list for every layer
        self.channels = [in_channels, 5, 10, 20, 50, 1]

        # generate kernel size list
        self.kernels = [3, 5, 5, 3, 1]
        
        self.batchnorm = batchnorm
        if batchnorm:
            self.batchnorms = [1]*4 + [0]
        else:
            self.batchnorms = [0]*5

        assert len(self.dropouts) == depth
        assert len(self.channels) == depth + 1
        assert len(self.kernels) == depth
        assert len(self.batchnorms) == depth

        layers = []
        # layers = nn.ModuleList()

        # TODO fix notation depth layers?
        
        # deep layers
        for i in range(depth-1):
            layers.append(DVN_Block(
                self.channels[i],
                self.channels[i+1],
                self.kernels[i],
                self.batchnorms[i],
                self.dropouts[i]))
        # last layer
        layers.append(nn.Conv3d(self.channels[-2], self.channels[-1], self.kernels[-1])) 

        self.layers = nn.Sequential(*layers)

    def forward(self, x):

        return self.layers(x)

class DVN_Block(nn.Module):

    def __init__(self, in_size, out_size, kernel_size, batchnorm, dropout):
        super(DVN_Block, self).__init__()
        block = []

        # block = nn.ModuleList()
        
        block.append(nn.Conv3d(in_size, out_size, kernel_size))
        block.append(nn.ReLU())
        if batchnorm:
            block.append(nn.BatchNorm3d(out_size))
        if dropout:
            block.append(nn.Dropout3d(dropout))
        
        self.block = nn.Sequential(*block)

    def forward(self, x):
        
        return self.block(x)











