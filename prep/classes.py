#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  6 16:44:10 2019

@author: sgerl
"""
from pathlib import Path
import scipy.ndimage
import os

import scipy.io as sio
from scipy import interpolate
from scipy import ndimage
from scipy.optimize import minimize_scalar

import imageio

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import time
import warnings

import nibabel as nib

from skimage import exposure
from skimage import morphology
from skimage import transform


# CLASS FOR ONE DATASET
class RSOM():
    """
    class for preparing RSOM matlab data for layer and vessel segmentation
    """
    def __init__(self, filepathLF, filepathHF, filepathSURF='none'):
        """
        Create empty instance of RSOM
        """
        # if the filepaths are strings, generate PosixPath objects
        filepathLF = Path(filepathLF)
        filepathHF = Path(filepathHF)
        filepathSURF = Path(filepathSURF)

        # extact datetime number
        print(filepathLF.name)
        idx_1 = filepathLF.name.find('_')
        idx_2 = filepathLF.name.find('_', idx_1+1)
        DATETIME = filepathLF.name[idx_1:idx_2+1]
        
        # extract the 3 digit id + measurement string eg PAT001_RL01
        idxID = filepathLF.name.find('PAT')

        if idxID == -1:
            idxID = filepathLF.name.find('VOL')

        if idxID != -1:
            ID = filepathLF.name[idxID:idxID+11]
        else:
            # ID is different, extract string between Second "_" and third "_"
            # usually it is 6 characters long
            idx_3 = filepathLF.name.find('_', idx_2+1)
            ID = filepathLF.name[idx_2+1:idx_3]
        
        self.layer_end = None
        
        self.file = self.FileStruct(filepathLF, filepathHF, filepathSURF, ID, DATETIME)

    def prepare(self):

        self.read_matlab()
        self.flat_surface()
        self.cut_depth()
        self.norm_intensity()
        self.rescale_intensity()
        self.merge_volume_rgb()

    def read_matlab(self):
        '''
        read .mat files
        '''
        # load HF data
        self.matfileHF = sio.loadmat(self.file.HF)
        
        # extract high frequency Volume
        self.Vh = self.matfileHF['R']
        
        # load LF data
        self.matfileLF = sio.loadmat(self.file.LF)
        
        # extract low frequency Volume
        self.Vl = self.matfileLF['R']
        
        # load surface data
        try:
            self.matfileSURF = sio.loadmat(self.file.SURF)
        except:
            print(('WARNING: Could not load surface data, placing None type in'
                   'surface file. Method flatSURFACE is not going to be applied!!'))
            self.matfileSURF = None

    def save_matlab(self, output_dir, fstr='Crop'):
        sio.savemat(os.path.join(output_dir, os.path.basename(self.file.LF).replace('LF.mat', f'_{fstr}LF.mat')), self.matfileLF)
        sio.savemat(os.path.join(output_dir, os.path.basename(self.file.HF).replace('HF.mat', f'_{fstr}HF.mat')), self.matfileHF)

        
    def flat_surface(self):
        '''
        modify volumetric data in order to get a flat skin surface
        options:
            override = True. If False, Volumetric data of the unflattened
            Skin will be saved.
        '''
        if self.matfileSURF is not None:
            
            # parse surface data and dx and dy
            S = self.matfileSURF['surfSmooth']
            dx = self.matfileSURF['dx']
            dy = self.matfileSURF['dy']
            
            # create meshgrid for surface data
            xSurf = np.arange(0, np.size(S, 0)) * dx
            ySurf = np.arange(0, np.size(S, 1)) * dy
            xSurf -= np.mean(xSurf)
            ySurf -= np.mean(ySurf)
            xxSurf, yySurf = np.meshgrid(xSurf, ySurf)
        
            # create meshgrid for volume data
            # use grid step dv
            # TODO: extract from reconParams
            # TODO: solve problem: python crashes when accessing reconParams
            dv = 0.012
            xVol = np.arange(0, np.size(self.Vl, 2)) * dv
            yVol = np.arange(0, np.size(self.Vl, 1)) * dv
            xVol -= np.mean(xVol)
            yVol -= np.mean(yVol)
            xxVol, yyVol = np.meshgrid(xVol, yVol)
        
            # generate interpolation function
            fn = interpolate.RectBivariateSpline(xSurf, ySurf, S)
            Sip = fn(xVol, yVol)
            
            Sip -= np.mean(Sip)
            
            # flip, to fit the grid
            Sip = Sip.transpose()
            
            self.Sip = Sip
            
            # for every surface element, calculate the offset
            # and shift volume elements perpendicular to surface
            for i in np.arange(np.size(self.Vl, 1)):
                for j in np.arange(np.size(self.Vl, 2)):
                    
                    offs = int(-np.around(Sip[i, j]/2))
                    
                    self.Vl[:, i, j] = np.roll(self.Vl[:, i, j], offs)
                    self.Vh[:, i, j] = np.roll(self.Vh[:, i, j], offs)
                    
                    # replace values rolled inside epidermis with zero
                    if offs < 0:
                        self.Vl[offs:, i, j] = 0
                        self.Vh[offs:, i, j] = 0
        
    def plot_surface(self):
        '''
        plot the surfaceData used for the normalization
        mainly for debugging purposes
        '''
        
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        
        # parse surface data and dx and dy
        S = self.matfileSURF['surfSmooth']
        dx = self.matfileSURF['dx']
        dy = self.matfileSURF['dy']
        
        xSurf = np.arange(0, np.size(S, 0)) * dx
        ySurf = np.arange(0, np.size(S, 1)) * dy
        xSurf -= np.mean(xSurf)
        ySurf -= np.mean(ySurf)
        xxSurf, yySurf = np.meshgrid(xSurf, ySurf)
        
        S = S - np.amin(S)

        surf = ax.plot_surface(xxSurf, yySurf, S.transpose(), cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.show()

    def sliding_window_mip(self, size, axis=1):
        """

        Args:
            size:
            axis: 1 or 2

        Returns:

        """
        self.Vl_1 = scipy.ndimage.maximum_filter1d(self.Vl_1, size, axis=axis)
        self.Vh_1 = scipy.ndimage.maximum_filter1d(self.Vh_1, size, axis=axis)
        
    def norm_intensity(self):
        '''
        normalize intensities to [0 1]
        '''
        
        self.Vl_1 = np.true_divide(self.Vl, np.amax(self.Vl))
        self.Vh_1 = np.true_divide(self.Vh, np.amax(self.Vh))
            
        self.Vl_1[self.Vl_1 > 1] = 1
        self.Vh_1[self.Vh_1 > 1] = 1
        self.Vl_1[self.Vl_1 < 0] = 0
        self.Vh_1[self.Vh_1 < 0] = 0
        
    def rescale_intensity(self, reduced_intensity=False):
        '''
        rescale intensities, quadratic transform, crop values
        '''
        print(f"{reduced_intensity=}")
        if not reduced_intensity:
            self.Vl_1 = exposure.rescale_intensity(self.Vl_1, in_range = (0, 0.2))
            self.Vh_1 = exposure.rescale_intensity(self.Vh_1, in_range = (0, 0.1))
        else:
            self.Vl_1 = exposure.rescale_intensity(self.Vl_1, in_range = (0, 0.3))
            self.Vh_1 = exposure.rescale_intensity(self.Vh_1, in_range = (0, 0.15))
        
        self.Vl_1 = self.Vl_1**2
        self.Vh_1 = self.Vh_1**2

        if not reduced_intensity:
            self.Vl_1 = exposure.rescale_intensity(self.Vl_1, in_range = (0.05, 1))
            self.Vh_1 = exposure.rescale_intensity(self.Vh_1, in_range = (0.02, 1))
        else:
            self.Vl_1 = exposure.rescale_intensity(self.Vl_1, in_range = (0.03, 1))
            self.Vh_1 = exposure.rescale_intensity(self.Vh_1, in_range = (0.02, 1))

    def calc_mip(self, axis = 1, do_plot = True, cut_z=0):
        '''
        plot maximum intensity projection along specified axis
        options:
            axis = 0,1,2. Axis along which to project
            do_plot = True. Plot into figure
            cut     = cut from volume in z direction
                      needed for on top view without epidermis
            
        '''
        axis = int(axis)
        if axis > 2:
            axis = 2
        if axis < 0:
            axis = 0
            
        # maximum intensity projection
        self.Pl = np.amax(self.Vl[cut_z:,...], axis = axis)
        self.Ph = np.amax(self.Vh[cut_z:,...], axis = axis)
        
        # calculate alpha
        res = minimize_scalar(self.calc_alpha, bounds=(0, 100), method='bounded')
        alpha = res.x
        
        self.P = np.dstack([self.Pl, alpha * self.Ph, np.zeros(self.Ph.shape)])
        
        # cut negative values, in order to allow rescale to uint8
        self.P[self.P < 0] = 0
        
        self.P = exposure.rescale_intensity(self.P, out_range = np.uint8)
        self.P = self.P.astype(dtype=np.uint8)
        
        # rescale intensity
        val = np.quantile(self.P, (0.8, 0.9925))
        
        self.P = exposure.rescale_intensity(self.P, 
                                            in_range = (val[0], val[1]), 
                                            out_range = np.uint8)
        if do_plot:
            plt.figure()
            plt.imshow(self.P)
            plt.title(str(self.file.ID))
            plt.show()
    
    def calc_alpha(self, alpha):
        '''
        MIP helper function
        '''
        return np.sum(np.square(self.Pl - alpha * self.Ph))
    
    def calc_alpha_3d(self, alpha):
        '''
        MIP helper function
        '''
        return np.sum(np.square(self.Vl_split - alpha * self.Vh_split))
    
    def calc_mip3d(self, do_plot = True):
        '''
        plot maximum intensity projection along second axis
        options:
            do_plot = True
        '''
    
        # split along axis=1, 171 pixel / 9 = 19
        # get back 9 equally sized arrays
        self.Vl_split = np.split(self.Vl, 9, axis = 1)
        self.Vh_split = np.split(self.Vh, 9, axis = 1)
        
        # for every slice, perform a maximum intensity projection
        for idx in range(len(self.Vl_split)):
            self.Vl_split[idx] = np.amax(self.Vl_split[idx], axis = 1, keepdims = True)
            self.Vh_split[idx] = np.amax(self.Vh_split[idx], axis = 1, keepdims = True)
        
        self.Vl_split = np.concatenate(self.Vl_split, axis = 1)
        self.Vh_split = np.concatenate(self.Vh_split, axis = 1)
            
        # global alpha calculation (all slices)
        res = minimize_scalar(self.calc_alpha_3d, bounds=(0, 100), method='bounded')
        alpha = res.x
        
        # RGB stack
        self.P_sliced = np.stack([self.Vl_split, 
                                  alpha * self.Vh_split, 
                                  np.zeros(self.Vl_split.shape)], axis = -1)
        
        self.P_sliced[self.P_sliced < 0] = 0
        self.P_sliced = exposure.rescale_intensity(self.P_sliced, out_range = np.uint8)
        self.P_sliced = self.P_sliced.astype(dtype=np.uint8)
        
        # rescale intensity
        val = np.quantile(self.P_sliced,(0.8, 0.9925))
        
        self.P_sliced = exposure.rescale_intensity(self.P_sliced, 
                                                  in_range = (val[0], val[1]), 
                                                  out_range = np.uint8)
        if do_plot:
            shp = self.P_sliced.shape
            P_ = self.P_sliced.copy()
            P_[:,:,-3:-1,:] = 255
            plt.figure()
            plt.imshow(P_.reshape((shp[0], shp[1]*shp[2], -1)))
            plt.title(str(self.file.ID))
            plt.show()
        
    def merge_volume_rgb(self):
        '''
        merge low frequency and high frequency data feeding into different
        colour channels
        '''
        B = np.zeros(np.shape(self.Vl_1))
        
        self.Vm = np.stack([self.Vl_1, self.Vh_1, B], axis = -1)
        
    def _debug_cut_empty_or_layer(self, dest):
        ''' 1D projection along z '''
        
        
        proj = np.sum(self.Vl_1, axis=(1,2))
        proj *= 1/proj.max()
        
        # debug
        fig, ax = plt.subplots()
        x = np.arange(self.layer_end,500)
        ax.plot(x, proj)
        
        # mark where last nonzero value is
        last_nz = np.nonzero(proj)[0][-1]


        # debug
        print('last nonzero index', last_nz + self.layer_end)
        ax.plot([last_nz + + self.layer_end, last_nz + self.layer_end], [-1, 1])
        
        lenproj = len(proj)
        
        proj = proj[:min(last_nz + 25, lenproj - 1)]
        
        #debug
        x = x[:min(last_nz + 25, lenproj - 1)]
        print(x.shape, proj.shape)
        
        filter_size = min([50, 25 + max([0, (len(proj) - 220)/7])])
        print(filter_size)
        # TODO:
        # dynamically adapt filter sizes
        # verify all steps
        
        
        # gaussian filter
        proj_f = ndimage.gaussian_filter1d(proj.copy(), filter_size)
        proj_f *= 1/proj_f.max()
    
        # debug
        ax.plot(x, proj_f)
        
        d_proj_f = np.gradient(proj_f)
        d_proj_f *= 1/np.amax(np.abs(d_proj_f))
        
        # debug
        ax.plot(x, d_proj_f)
        
        # find the locations of d_proj_f == 0
        ax.plot(x, np.gradient(np.sign(d_proj_f)))
        nz_idx = np.nonzero(np.gradient(np.sign(d_proj_f)))[0]
        print('nz_idx       ', nz_idx)
        
        # extract unitary locations
        # i.e. remove subsequent values if they have distance 1
        nz_idx_ = []
        for idx in range(len(nz_idx)):
            if not idx:
                nz_idx_.append(nz_idx[idx])
            else:
                if not nz_idx[idx] == nz_idx[idx-1] + 1:
                    nz_idx_.append(nz_idx[idx])
                    
        print('nz_idx_cleaned',nz_idx_)
        
        # check if extremum is too close to epidermis    
        nz_idx__ = []
        for el in nz_idx_:
            if el <= len(proj) / 5:
                warnings.warn('Strange distribution in z-direction detected.', UserWarning)
            else:
                nz_idx__.append(el)
                
        nz_idx_ = nz_idx__

        # we have at least 2 extrema
        if len(nz_idx_) >= 2:
            print('we have at least 2 extremum')
            dd_proj_f = np.gradient(d_proj_f)[nz_idx_]
            if dd_proj_f[-1] > 0:
                print('last one is a minimum, probably no noise.')
            else:
                proj_f_fine = ndimage.gaussian_filter1d(proj.copy(), filter_size/5)
                ax.plot(x, proj_f_fine)
                
                # maximum in "lower" part is at least 1/3 of global maximum
                if 3 * proj_f_fine[nz_idx_[-2]:].max() >= proj_f_fine.max():
                    extremum_scale = max(proj_f[nz_idx_[-2]], proj_f[nz_idx_[-1]])
                    
                    if proj_f[nz_idx_[-2]]/extremum_scale < 0.9 * proj_f[nz_idx_[-1]]/extremum_scale:  
                        
                        print('most likely layer noise found')
                        proj_f_fine *= 1/proj_f_fine[nz_idx_[-2]:nz_idx_[-1]].max()
                        
                        # cut out part of interest
                        x_ = np.arange(self.layer_end + nz_idx_[-2], 
                                       self.layer_end + nz_idx_[-1])
                        proj_f_fine_ = proj_f_fine[nz_idx_[-2]:nz_idx_[1]]
                        
                        proj_f_fine_[proj_f_fine_ < 0.8 * proj_f_fine_.max()] = 0
                        cutoff_idx = self.layer_end + nz_idx_[-2] + np.nonzero(proj_f_fine_)[0][0]
                        ax.plot([cutoff_idx, cutoff_idx], [-1, 1])
                        
                        self.P[cutoff_idx,:] = 255
                    else:
                        print('extrema not significant enough')
                else:
                    print('intensity of last minimum too low.')
                

        #ax.set(xlabel='index', ylabel='intensity')
        ax.grid(True, which='both')
        ax.minorticks_on()
        ax.set_xlim(left=0,right=500)
        ax.tick_params(axis="y",direction="in", pad=-22)
        ax.tick_params(axis="x",direction="in", pad=-15)
       
        fig.tight_layout()
        fig.canvas.draw()
        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        
        print(image_from_plot.shape)
        
        # merge with mip
        image_from_plot = np.rot90(np.rot90(np.rot90(image_from_plot)))
        image_from_plot = transform.resize(image_from_plot, self.P.shape
                                           + np.array([40,0,0]))
        
        P = np.pad(self.P, ((20,20),(0,0),(0,0)),
                   mode='constant',
                   constant_values=127)
        
        P = np.concatenate((P.astype(np.uint8), (255*image_from_plot).astype(np.uint8)),axis=1)
        #print(P.dtype, image_from_plot.dtype)
        #print(P.max(), image_from_plot.max())
        
        imageio.imwrite(os.path.join(dest, self.file.ID + '.png'), P)

    def cut_empty_or_layer(self):
        ''' 1D projection along z '''
        
        proj = np.sum(self.Vl_1, axis=(1,2))
        proj *= 1/proj.max()
        
        # mark where last nonzero value is
        last_nz = np.nonzero(proj)[0][-1]
        
        # assign default cut of at last occurence of values not zero
        cutoff_idx = last_nz
          
        proj = proj[:min(last_nz + 25, len(proj) - 1)]
        
        filter_size = min([50, 25 + max([0, (len(proj) - 220)/7])])

        # gaussian filter
        proj_f = ndimage.gaussian_filter1d(proj.copy(), filter_size)
        proj_f *= 1/proj_f.max()
            
        d_proj_f = np.gradient(proj_f)
        d_proj_f *= 1/np.amax(np.abs(d_proj_f))
                
        # find the locations of d_proj_f == 0
        nz_idx = np.nonzero(np.gradient(np.sign(d_proj_f)))[0]

        # extract unitary locations
        # i.e. remove subsequent values if they have distance 1
        nz_idx_ = []
        for idx in range(len(nz_idx)):
            if not idx:
                nz_idx_.append(nz_idx[idx])
            else:
                if not nz_idx[idx] == nz_idx[idx-1] + 1:
                    nz_idx_.append(nz_idx[idx])
        
        # check if extremum is too close to epidermis    
        nz_idx__ = []
        for el in nz_idx_:
            if el <= len(proj) / 5:
                warnings.warn('Strange distribution in z-direction detected.', UserWarning)
            else:
                nz_idx__.append(el)
                
        nz_idx_ = nz_idx__

        # we have at least 2 extrema
        if len(nz_idx_) >= 2:
            dd_proj_f = np.gradient(d_proj_f)[nz_idx_]
            if dd_proj_f[-1] > 0:
                pass
            else:
                proj_f_fine = ndimage.gaussian_filter1d(proj.copy(), filter_size / 5)
                
                # maximum in "lower" part is at least 1/3 of global maximum
                if 3 * proj_f_fine[nz_idx_[-2]:].max() >= proj_f_fine.max():
            
                    # extremum is significant, and not almost saddle point
                    extremum_scale = max(proj_f[nz_idx_[-2]], proj_f[nz_idx_[-1]])
                    if proj_f[nz_idx_[-2]]/extremum_scale < 0.9 * proj_f[nz_idx_[-1]]/extremum_scale: 
                        proj_f_fine *= 1 / proj_f_fine[nz_idx_[-2]:nz_idx_[-1]].max()
                        
                        # cut out part of interest
                        proj_f_fine_ = proj_f_fine[nz_idx_[-2]:nz_idx_[1]]
                        
                        proj_f_fine_[proj_f_fine_ < 0.8 * proj_f_fine_.max()] = 0
                        
                        # override cutoff idx
                        cutoff_idx = nz_idx_[-2] + np.nonzero(proj_f_fine_)[0][0]
                        print('Layer noise found!')
                        
        
        # cut away
        print('cut_empty_or_layer method, cutting at', cutoff_idx + self.layer_end)
        
        print(self.Vl_1.shape, cutoff_idx)
        self.Vl_1 = self.Vl_1[:cutoff_idx + 1, :, :]
        self.Vh_1 = self.Vh_1[:cutoff_idx + 1, :, :]
        
        print(self.Vl_1.shape)
        
        self.vessel_end = cutoff_idx + self.layer_end
        
    def cut_empty_or_layer_manual(self, path, fstr='manual'):
        '''
        cut off the epidermis with loading corresponding segmentation mask.
        '''
        print('cutLayer method')
        
        # generate path
        filename = 'R' + self.file.DATETIME + self.file.ID + '_' + fstr
        file = os.path.join(path, filename)
        
        print('Loading', file)
        f = open(file)
        cutoff_idx = int(str(f.read()))
        f.close()
        
        if cutoff_idx == -1:
            proj = np.sum(self.Vl_1, axis=(1,2))
            proj *= 1/proj.max()
        
            # mark where last nonzero value is
            cutoff_idx = np.nonzero(proj)[0][-1]
            
            
        
        print('Cutting at', cutoff_idx)
        
        # cut away
        self.Vl_1 = self.Vl_1[:cutoff_idx + 1, :, :]
        self.Vh_1 = self.Vh_1[:cutoff_idx + 1, :, :]
        
        self.vessel_end = cutoff_idx + self.layer_end
        
    def cut_depth(self):
        '''
        cut Vl and Vh to 500 x 171 x 333
        '''
        zmax = 500
        
        # extract shape
        shp = self.Vl.shape
        
        if shp[0] >= zmax:
            self.Vl = self.Vl[:500,:,:]
            self.Vh = self.Vh[:500,:,:]
        else:
            ext = zmax - shp[0]
            print('Extending volume. old shape:', shp)
            
            self.Vl = np.concatenate([self.Vl, np.zeros((ext, shp[1], shp[2]))], axis = 0)
            self.Vh = np.concatenate([self.Vh, np.zeros((ext, shp[1], shp[2]))], axis = 0)  
            
            print('New shape:', self.Vl.shape)

    def save_mip(self, destination, fstr = '', scale=1):
        '''
        save MIP as 2d image
        '''
        
        if scale != 1:
            P = transform.rescale(self.P, scale, order=3, multichannel=True)
            # strangely transform.rescale is not dtype consistent?
            P = exposure.rescale_intensity(P, out_range=np.uint8)
            P = P.astype(np.uint8)
            
            # this was for mip example for thesis
            #from skimage.filters import unsharp_mask
            #P = unsharp_mask(P, radius=10, amount=1)
            #print(P.max())
            #P = (P**1.1)*0.7
            #P = P[:350*4,...]
            #P = P.astype(np.uint8)
        else:
            P = self.P
            
         # generate Path object
        destination = Path(destination)
        
        # generate filename
        img_file = (destination / ('R' + 
                                   self.file.DATETIME + 
                                   self.file.ID +
                                   '_' +
                                   fstr +
                                   '.png')).resolve()
        print(str(img_file))
        
        
        imageio.imwrite(img_file, P)

    def save_surface(self, destination, fstr = ''):

        '''
        save surface as 2d image with colorbar?
        '''
        plt.ioff()
        plt.figure()
        
        Surf = self.Sip-np.amin(self.Sip)
        
        plt.imshow(Surf, cmap=cm.jet)
        plt.colorbar()
        #plt.show()
        
        
        
        
        # generate Path object
        destination = Path(destination)
        
        # generate filename
        img_file = (destination / ('R' + 
                                   self.file.DATETIME + 
                                   self.file.ID +
                                   '_' +
                                   fstr +
                                   '.png')).resolve()
        print(str(img_file))
        
        plt.title(str(self.file.ID))
        plt.savefig(img_file)
        
        plt.close() 
    
        
        #imageio.imwrite(img_file, self.Sip)
        
    def save_mip3d(self, destination, fstr = ''):
        ''' save rgb maximum intensity projection volume'''
        
        # Vm is a 4-d numpy array, with the last dim holding RGB
        shape_3d = self.P_sliced.shape[0:3]
        rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
        #self.Vm = self.P_sliced.astype('u1')
        self.P_sliced = self.P_sliced.copy().view(rgb_dtype).reshape(shape_3d)
        img = nib.Nifti1Image(self.P_sliced, np.eye(4))
        
        # generate Path object
        destination = Path(destination)
        
        # generate filename
        nii_file = (destination / ('R' + 
                                   self.file.DATETIME + 
                                   self.file.ID +
                                   '_' +
                                   fstr +
                                   '.nii.gz')).resolve()
        print(str(nii_file))
        nib.save(img, str(nii_file))

    def save_raw_volumes(self, destination):
        img_l = nib.Nifti1Image(self.Vl, np.eye(4))
        img_h = nib.Nifti1Image(self.Vh, np.eye(4))

        nii_file = os.path.join(destination, 'R' + self.file.DATETIME + self.file.ID + '_raw_')

        nib.save(img_l, nii_file + 'LF.nii.gz')
        nib.save(img_h, nii_file + 'HF.nii.gz')

    def save_volume(self, destination, fstr = ''):
        '''
        save rgb volume
        '''
        
        self.Vm = exposure.rescale_intensity(self.Vm, out_range = np.uint8)
        
        # Vm is a 4-d numpy array, with the last dim holding RGB
        shape_3d = self.Vm.shape[0:3]
        rgb_dtype = np.dtype([('R', 'u1'), ('G', 'u1'), ('B', 'u1')])
        self.Vm = self.Vm.astype('u1')
        self.Vm = self.Vm.copy().view(rgb_dtype).reshape(shape_3d)
        img = nib.Nifti1Image(self.Vm, np.eye(4))
        
        
        # generate Path object
        destination = Path(destination)
        
        # if this is a cut file, need to construct the cut z-value
        # this is only for vesnet preparation
        if self.layer_end is not None:
            z_cut = '_' + 'z' + str(self.layer_end)
        else:
            z_cut = ''
        
        # generate filename
        nii_file = (destination / ('R' + 
                                   self.file.DATETIME + 
                                   self.file.ID +
                                   z_cut +
                                   '_' +                                   
                                   fstr +
                                   '.nii.gz')).resolve()
        print(str(nii_file))

        
        nib.save(img, str(nii_file))
        
    class FileStruct():
        def __init__(self, filepathLF, filepathHF, filepathSURF, ID, DATETIME):
            self.LF = filepathLF
            self.HF = filepathHF
            self.SURF = filepathSURF
            self.ID = ID
            self.DATETIME = DATETIME
    
    @staticmethod
    def loadNII(path):
        img = nib.load(path)
        return img.get_fdata()

class RSOM_vessel(RSOM):
    '''
    additional methods for preparing RSOM data for vessel segmentation,
    e.g. cut away epidermis
    '''

    def prepare(self, path, mode='pred', fstr='pred.nii.gz'):
        self.read_matlab()
        self.flat_surface()
        self.cut_depth()
        self.cut_layer(path, mode=mode, fstr=fstr)
        self.norm_intensity()
        self.rescale_intensity()
        self.merge_volume_rgb()
    
    def mask_layer(self, path, mode='pred', fstr='layer_pred.nii.gz'):
        '''
        cut off the epidermis with loading corresponding segmentation mask.
        '''
        
        # generate path
        filename = 'R' + self.file.DATETIME + self.file.ID + '_' + fstr
        file = os.path.join(path, filename)
        
        print('Loading', file)
        
        # two modes supported, extract from prediction volume
        # or manual input through file   
        if mode == 'pred':
            
            img = nib.load(file)
            self.S = img.get_fdata()
            self.S = self.S.astype(np.uint8)
            
            assert self.Vl.shape == self.S.shape, 'Shapes of raw and segmentation do not match'
            
            print(self.Vl.shape)
            print(self.S.shape)
            
            
        
        # cut away
        self.Vl[self.S.astype(np.bool)] = 0
        self.Vh[self.S.astype(np.bool)] = 0
        print('after masking')
        for x in np.arange(self.S.shape[1]):
            for y in np.arange(self.S.shape[2]):
                nz = np.nonzero(self.S[:, x, y])
                # print(nz)
                nz = nz[0]
                if len(nz) > 0:
                    upper = nz[0]
                    self.Vl[:upper, x, y] = 0
                    self.Vh[:upper, x, y] = 0
        self.layer_end = 0

    def cut_layer(self, path, mode='pred', fstr='layer_pred.nii.gz'):
        '''
        cut off the epidermis with loading corresponding segmentation mask.
        '''
        
        # generate path
        filename = 'R' + self.file.DATETIME + self.file.ID + '_' + fstr
        file = os.path.join(path, filename)
        
        print('Loading', file)
        
        # two modes supported, extract from prediction volume
        # or manual input through file   
        if mode == 'pred':
            
            img = nib.load(file)
            self.S = img.get_fdata()
            self.S = self.S.astype(np.uint8)
            
            assert self.Vl.shape == self.S.shape, 'Shapes of raw and segmentation do not match'
            
            print(self.Vl.shape)
            print(self.S.shape)
            
            
            # for every slice in x-y plane, calculate label sum
            label_sum = np.sum(self.S, axis=(1, 2))
            
            max_occupation = np.amax(label_sum) / (self.S.shape[1] * self.S.shape[2])
            max_occupation_idx = np.argmax(label_sum)
            
            print('Max occ', max_occupation)
            print('idx max occ', max_occupation_idx)
            if max_occupation >= 0.01:
                # normalize
                label_sum = label_sum.astype(np.double) / np.amax(label_sum)
                
                # define cutoff parameter
                cutoff = 0.05
                
                label_sum_bin = label_sum > cutoff
                     
                label_sum_idx = np.squeeze(np.nonzero(label_sum_bin))
            
                layer_end = label_sum_idx[-1]
                
                # additional fixed pixel offset
                offs = 10
                layer_end += offs
            else:
                print("WARNING:  Could not determine valid epidermis layer.")
                layer_end = 0
    
        elif mode == 'manual':
            f = open(file)
            layer_end = int(str(f.read()))
            f.close()
        else:
            raise NotImplementedError
        
        print('Cutting at', layer_end)
        
        # cut away
        self.Vl = self.Vl[layer_end:, :, :]
        self.Vh = self.Vh[layer_end:, :, :]
        
        self.layer_end = layer_end

    def backgroundAnnot_replaceVessel(self, path, mode='manual', fstr='ves_cutoff'):
        '''
        cut off the epidermis with loading corresponding segmentation mask.
        '''
        
        # generate path
        filename = 'R' + self.file.DATETIME + self.file.ID + '_' + fstr
        file = os.path.join(path, filename)
        
        print('Loading', file)
        
        if mode=='manual':
            f = open(file)
            background_end = int(str(f.read()))
            f.close()
            
        print('Replace with 0: z<=', background_end)

        
        # shift, in case layer was cut before,
        # may rise error if self.layer_end undefined
        background_end = background_end - self.layer_end
        
        # replace with 0
        self.Vl_1[:background_end, :, :] = 0
        self.Vh_1[:background_end, :, :] = 0
        
        # cut unneccessary background
        # but keep 200 pixels extension to get a similar volume size
        if background_end - 200 >= 0:
            cut = background_end - 200
        else:
            cut = 0
        #self.Vl_1 = self.Vl_1[cut:, :, :]
        #self.Vh_1 = self.Vh_1[cut:, :, :]
        
    def rescale_intensity(self):
        '''
        overrides method in class RSOM, because vessel segmentation needs 
        different rescale
        '''
        print('Vessel rescaleINTENSITY method')
        self.Vl_1 = exposure.rescale_intensity(self.Vl_1, in_range = (0, 0.25))
        self.Vh_1 = exposure.rescale_intensity(self.Vh_1, in_range = (0, 0.15))
        #self.Vl_1[:,:,:] = 0
            
        self.Vl_1 = self.Vl_1**2
        self.Vh_1 = self.Vh_1**2
            
        self.Vl_1 = exposure.rescale_intensity(self.Vl_1, in_range = (0.05, 1))
        self.Vh_1 = exposure.rescale_intensity(self.Vh_1, in_range = (0.05, 1))
        
    def threshold_segmentation(self):
        
        self.Vseg = np.logical_or(np.logical_or((self.Vh_1 + self.Vl_1) >= 1, 
                                                self.Vl_1 > 0.3),
                                    self.Vh_1 > 0.7)
        
        # hack for background annotation. zeros only
        # self.Vseg = np.zeros_like(self.Vh_1)
        return self.Vseg
        
    def math_morph(self):
        
        # TODO PADDING AND REMOVE 
        # probably the problem is in binary_closing??

        
        filter = ndimage.morphology.generate_binary_structure(3,1).astype(np.int64)
        print(filter)
        M = ndimage.convolve(self.Vseg.astype(np.int64), filter, mode='constant',cval=0)
        
        # fill holes custom function
        holesMask = np.logical_or(M == 6, M == 5)
        self.Vseg = np.logical_or(self.Vseg, holesMask)
        
        # remove single pixels or only 2 pixels
        singleMask = M >= 2
        self.Vseg = np.logical_and(singleMask, self.Vseg)

        # closing
        self.Vseg = (ndimage.morphology.binary_closing(
                np.pad(self.Vseg, 1, mode='edge')))[1:-1, 1:-1, 1:-1]
        
        # remove small objects
        # 30 seems a good value, tested from 10 to 40
        self.Vseg = morphology.remove_small_objects(self.Vseg, 30)
        
    def save_segmentation(self, destination, fstr='th'):
        '''
        save rgb volume
        '''
    
        Vseg = self.Vseg.astype(np.uint8)
        img = nib.Nifti1Image(Vseg, np.eye(4))
        
        if self.layer_end is not None:
            z_cut = '_' + 'z' + str(self.layer_end)
        else:
            z_cut = ''
        
        path = os.path.join(destination,
                            'R' + 
                            self.file.DATETIME + 
                            self.file.ID +
                            z_cut +
                            '_' +                                   
                            fstr +
                            '.nii.gz')
                            
        
        nib.save(img, path)
        
    def save_volume_float(self, destination, fstr = ''):
        '''
        override method from RSOM class
        save rgb volume
        '''
        
        #self.Vm = exposure.rescale_intensity(self.Vm, out_range = np.uint8)
        
        # Vm is a 4-d numpy array, with the last dim holding RGB
        #dtype_ = 'u1'
        #dtype_ = 'f4'  # np.float32
        
        #shape_3d = self.Vm.shape[0:3]
        #rgb_dtype = np.dtype([('R','f4'), ('G', 'f4'), ('B', 'f4')])
        self.Vm = self.Vm.astype(np.float32)
        
        #self.Vm = self.Vm.copy().view(rgb_dtype).reshape(shape_3d)
        
        V_R = self.Vm[...,0]
        V_G = self.Vm[...,1]
        
        
        for V, c in [(V_R, 'R'), (V_G, 'G')]:
        
            img = nib.Nifti1Image(V, np.eye(4))
            
            
            # generate Path object
            destination = Path(destination)
            
            # if this is a cut file, need to construct the cut z-value
            # this is only for vesnet preparation
            if self.layer_end is not None:
                z_cut = '_' + 'z' + str(self.layer_end)
            else:
                z_cut = ''
            
            # generate filename
            nii_file = (destination / ('R' + 
                                       self.file.DATETIME + 
                                       self.file.ID +
                                       z_cut +
                                       '_' +                                   
                                       fstr +
                                       '_' +
                                       c +
                                       '.nii.gz')).resolve()
            print(str(nii_file))
        
            
            nib.save(img, str(nii_file))
            
class RSOM_mip_interp():
    def __init__(self, filepath):
        ''' create an instance of RSOM_MIP_IP; requires path '''
        self.filepath = filepath
        
    def readNII(self):
        ''' read in the .nii.gz files specified in self.filepath'''
        
        img = nib.load(str(self.filepath))
        self.L_sliced = img.get_fdata()
        self.L_sliced = self.L_sliced.astype(np.uint8)
        
    def saveNII(self, destination, fstr = ''):
        ''' '''
        

        self.L = self.L.astype(np.uint8)
        img = nib.Nifti1Image(self.L, np.eye(4))
        
        file = self.filepath
        
        name = file.name
        
        name = name.replace('mip3d_l.nii.gz',('l' + fstr + '.nii.gz'))
        
            # generate Path object
        destination = Path(destination)
        
        # generate filename
        nii_file = destination / name
        print(str(nii_file))
        nib.save(img, str(nii_file))
        
    def interpolate(self):
        ''' interpolate the .nii.gz label files along axis=1 '''
        
        # label volume is           depth x 9 x 333
        # need to interpolate to    depth x 171 x 333
        
        x_mip = 9
        x_rep = 171
        
        label_min = np.amin(self.L_sliced)
        label_max = np.amax(self.L_sliced)
        
        #self.L_sliced = exposure.rescale_intensity(self.L_sliced, in_range = (label_min, label_max), out_range = np.uint8)
        #self.L_sliced = self.L_sliced.astype(np.uint8)
        
        shp = self.L_sliced.shape
    
        # create grid for L_sliced
        x1 = np.arange(shp[0])
        x2 = np.linspace(np.floor(x_rep/x_mip/2), x_rep - np.ceil(x_rep/x_mip/2), num = x_mip)
        x3 = np.arange(shp[2])
        
        # dimension to be interpolated
        x2_q = np.arange(x_rep)
        
        
        L_sliced_ = self.L_sliced.copy().astype(np.int64)
        
        # cut in direction dim 0 throught middle of the volume
        L_1d = L_sliced_[:,int(shp[1]/2),int(shp[2]/2)].copy()
        
        # find the number of labels, and their indices
        # np.unique returns the labels in ascending order
        labels, idx = np.unique(L_1d, return_index = True)
        
        n_labels = labels.size
        
        # TODO:
        # print('Warning. Processing data with poor implementation for label handling')
        
        for xx in np.arange(x_mip):
            for yy in np.arange(shp[2]):
            
               idx_nz = np.nonzero(L_sliced_[:, xx, yy])
               #print(idx_nz)
               idx_nz = idx_nz[0][-1]
               #print(idx_nz)
               L_sliced_[idx_nz+1:, xx, yy] = 2
               
               #print(np.unique(L_sliced_[:, xx, yy], return_index = True))
               
               #print(L_sliced_[idx_nz, xx, yy])
               
        # TODO: 
        #DANGEROUS HACK
        # print('warning, remove that dangerous hack')
        n_labels = n_labels + 1
        
        # check if labels are in shape: 0 1 2 3 4 already
                
        # INPUT: in dim 0: ascending index: label order: 0 1 2 3 4
        self.L = np.zeros((shp[0], x_rep, shp[2]))
        
        for nl in np.arange(n_labels - 1):
            surf = np.zeros((x_mip, shp[2]))
    
            for xx in np.arange(x_mip):
                for yy in np.arange(shp[2]):
                    
                    #idx = np.nonzero(np.logical_not(self.L_sliced[:, xx, yy]))
                    idx = np.nonzero(L_sliced_[:, xx, yy])
                    
                    surf[xx, yy] = idx[0][0]
                    
            fn = interpolate.interp2d(x3, x2, surf, kind = 'cubic')
            surf_ip = fn(x3, x2_q)
            
            for xx in np.arange(x_rep):
                for yy in np.arange(shp[2]):
                    self.L[0:np.round(surf_ip[xx, yy]).astype(np.int), xx, yy] += 1
            
            # NEXT LABEL
            L_sliced_ -= 1
            L_sliced_[L_sliced_ < 0] = 0
            
        # TODO   
        # print('Warning. Processing data with poor implementation for label handling')
        self.L[self.L == 2] = 0
        
            
        return self.L
    
        

class RsomVisualization(RSOM):
    '''
    subclass of RSOM
    for various visualization tasks
    '''

    def load_seg(self, filename):
        '''
        load segmentation file (can be labeled our predicted)

        '''
        return (self.loadNII(filename)).astype(np.uint8)

    def calc_mip_ves_seg(self, seg, axis=1, padding=(0, 0)):
        '''
        calculate pseudo-mip of boolean segmentation
        '''
        seg = self.load_seg(seg)

        mip = np.sum(seg, axis=axis) >= 1

        # probably need some further adjustment on mip.shape[-1]
        print('seg mip shape:', mip.shape)
        mip = np.concatenate((np.zeros((padding[0], mip.shape[-1]), dtype=np.bool), 
                              mip.astype(np.bool),
                              np.zeros((padding[1], mip.shape[-1]), dtype=np.bool)), axis=0)
        # naming convention self.P is normal MIP 
        self.axis_P = axis
        self.P_seg = mip

    def calc_mip_lay_seg(self, seg, axis=1, padding=(0, 0)):

        seg = self.load_seg(seg)

        mip = np.sum(seg, axis=axis) >= 1

        print('seg mip shape:', mip.shape)
        mip = np.concatenate((np.zeros((padding[0], mip.shape[-1]), dtype=np.bool), 
                              mip.astype(np.bool),
                              np.zeros((padding[1], mip.shape[-1]), dtype=np.bool)), axis=0)
        # naming convention self.P is normal MIP 
        self.axis_P = axis
        self.P_seg = mip

    def merge_mip_ves(self, do_plot=True):
        '''
        merge MIP and MIP of segmentation with feeding into blue channel
        '''

        P_seg = self.P_seg.astype(np.float32)
        P_seg_edge = filters.sobel(P_seg)
        P_seg_edge = P_seg_edge/np.amax(P_seg_edge)

        # feed into blue channel
        # blue = 150 * P_seg + 200 * P_seg_edge
        # very light edge
        blue = 150 * P_seg + 30 * P_seg_edge
        
        self.P_overlay = self.P.copy().astype(np.float32)
        self.P_overlay[:, :, 2] += blue
        self.P_overlay[self.P>255] = 255

        # test, white background
        # P = self.P.astype(np.int64)
        # P += 100*np.ones(P.shape, dtype=np.int64)

        # print(np.amax(P
        # A = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, scale]])
        # self.P = ndimage.affine_transform(self.P, A)

        # self.P[np.where((self.P == [0,0,0]).all(axis = 2))] = [255,255,255]   
        if do_plot:
            plt.figure()
            plt.imshow(self.P)
            plt.title(str(self.file.ID))
            #plt.imshow(P, aspect = 1/4)
            plt.show()
    
    def merge_mip_lay(self, do_plot=True):
        '''
        merge MIP and MIP of segmentation with feeding into blue channel
        '''

        P_seg = self.P_seg.astype(np.float32)
        P_seg_edge = filters.sobel(P_seg)
        P_seg_edge = P_seg_edge/np.amax(P_seg_edge)
        P_seg_edge = morphology.binary_dilation(P_seg_edge)
           

        # feed into blue channel
        # blue = 150 * P_seg + 200 * P_seg_edge
        # very light edge
        blue = 150 * P_seg + 30 * P_seg_edge
        blue[blue>255] = 255
       
        # P_seg_edge_ma = np.ma.masked_array(P_seg_edge,mask=P_seg_edge==0)
                
        self.P_overlay[:, :, 2] += 255*P_seg_edge
        self.P_overlay[:, :, 0] += 100*P_seg_edge
        self.P_overlay[:, :, 1] += 100*P_seg_edge
        self.P_overlay[self.P>255] = 255


        # test, white background
        # P = self.P.astype(np.int64)
        # P += 100*np.ones(P.shape, dtype=np.int64)

        # print(np.amax(P
        # A = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, scale]])
        # self.P = ndimage.affine_transform(self.P, A)

        # self.P[np.where((self.P == [0,0,0]).all(axis = 2))] = [255,255,255]   
        if do_plot:
            plt.figure()
            plt.imshow(self.P)
            plt.title(str(self.file.ID))
            #plt.imshow(P, aspect = 1/4)
            plt.show()

    def _rescale_mip(self, scale):

        self.P_overlay = self.P_overlay.astype(np.uint8)
        
        if scale != 1:
            
            self.P = transform.rescale(self.P, scale, order=3, multichannel=True)
            # strangely transform.rescale is not dtype consistent?
            self.P = exposure.rescale_intensity(self.P, out_range=np.uint8)
            self.P = self.P.astype(np.uint8)
            
            self.P_overlay = transform.rescale(self.P_overlay, scale, order=3, multichannel=True)
            # strangely transform.rescale is not dtype consistent?
            self.P_overlay = exposure.rescale_intensity(self.P_overlay, out_range=np.uint8)
            self.P_overlay = self.P_overlay.astype(np.uint8)
   
    def return_mip(self, scale=2):

        self._rescale_mip(scale)

        return self.P, self.P_overlay

    def save_comb_mip(self, dest, scale=2):
        
        self._rescale_mip(scale)
        
        if self.P.shape[0] > self.P.shape[1]:
            axis = 1
        else:
            axis = 0
        
        grey = 50
        
        img = np.concatenate((np.pad(self.P, 
                                     ((2, 2),(2, 2),(0, 0)), 
                                     mode='constant', 
                                     constant_values=grey),
                             np.pad(self.P_overlay, 
                                    ((2, 2),(2, 2),(0, 0)), 
                                    mode='constant',
                                     constant_values=grey)),
                             axis=axis)
        img = np.pad(img, 
                     ((2, 2),(2, 2),(0, 0)), 
                     mode='constant', 
                     constant_values=grey)
        
        img_file = os.path.join(dest, 'R' + 
                                   self.file.DATETIME + 
                                   self.file.ID +
                                   '_' +
                                   'combMIP_ax' +
                                   str(self.axis_P) + 
                                   '.png')
        print(str(img_file))
        
        
        imageio.imwrite(img_file, img)

class RsomVisualization_white(RsomVisualization):
    
    #def merge_mip_ves(self, do_plot=True):
    #    '''
    #    merge MIP and MIP of segmentation with feeding into blue channel
    #    '''
        
    #    self.P_overlay = _overlay(self.P, self.P_seg.astype(np.float32),
    #                              alpha=0.6)
        
    #    self.P_overlay[self.P>255] = 255
        
    #    if do_plot:
    #        plt.figure()
    #        plt.imshow(self.P)
    #        plt.title(str(self.file.ID))
    #        #plt.imshow(P, aspect = 1/4)
    #        plt.show()
    
    def merge_mip_lay(self, do_plot=True):
        '''
        merge MIP and MIP of segmentation with feeding into blue channel
        '''
        self.P_overlay = _overlay(self.P_overlay, self.P_seg.astype(np.float32),
                                  alpha=0.5)
        
        self.P_overlay[self.P>255] = 255
        
        if do_plot:
            plt.figure()
            plt.imshow(self.P)
            plt.title(str(self.file.ID))
            #plt.imshow(P, aspect = 1/4)
            plt.show()
