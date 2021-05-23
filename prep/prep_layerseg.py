#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 19:08:29 2019

@author: sgerl
"""


from pathlib import Path

import os
import sys

from classes import RSOM

if __name__ == '__main__':
    sys.path.append('../')

from utils.get_unique_filepath import get_unique_filepath

# define folder
#origin = '/home/stefan/Documents/RSOM/Diabetes/new_data/no_surf'
origin = '/home/stefan/RSOM/testing/mat'
# origin = '/media/nas_ads_mwn/AG-Ntziachristos/RSOM_Data/RSOM_Diabetes/Stefan/allmat'
# origin = '/media/nas_ads_mwn/AG-Ntziachristos/RSOM_Data/RSOM_Diabetes/Stefan/'

# destination = '/media/nas_ads_mwn/AG-Ntziachristos/RSOM_Data/RSOM_Diabetes/Stefan/'
# destination = '/home/sgerl/Documents/PYTHON/TestDataset20190411/selection/other_preproccessing_tests/sliding_mip_6'
destination = '/home/stefan/RSOM/testing/mat/out'


# mode
mode = 'dir'

if mode=='dir':
    cwd = os.getcwd()
    # change directory to origin, and get a list of all files
    os.chdir(origin)
    all_files = os.listdir()
    os.chdir(cwd)
elif mode=='list':
    patterns = ['PAT052']
    all_files = [os.path.basename(get_unique_filepath(origin, pat)[0]) for pat in patterns]


# extract the LF.mat files,
filenameLF_LIST = [el for el in all_files if el[-6:] == 'LF.mat']

for idx, filenameLF in enumerate(filenameLF_LIST):
    # the other ones will be automatically defined
    filenameHF = filenameLF.replace('LF.mat','HF.mat')
    #TODO: review before running
    
    # extract datetime
    idx_1 = filenameLF.find('_')
    idx_2 = filenameLF.find('_', idx_1+1)
    filenameSurf = 'Surf' + filenameLF[idx_1:idx_2+1] + '.mat'
    
    
    # merge paths
    fullpathHF = (Path(origin) / filenameHF).resolve()
    fullpathLF = (Path(origin) / filenameLF).resolve()
    fullpathSurf = (Path(origin) / filenameSurf).resolve()
    
    Obj = RSOM(fullpathLF, fullpathHF, fullpathSurf)
    
    Obj.read_matlab()
    
    Obj.flat_surface()
    Obj.cut_depth()
    
    # surface for quick check
    #Obj.saveSURFACE((destination + ''), fstr = 'surf')
    
    # MIP image for quick check
    #Obj.calcMIP(do_plot = False)
    #Obj.saveMIP(destination, fstr = 'mip')
    
    # MIP 3D for annotation
    #Obj.calc_mip3d(do_plot = False)
    #Obj.save_mip3d(destination, fstr = 'mip3d')
    
    # VOLUME
    Obj.norm_intensity()
    Obj.rescale_intensity(dynamic_rescale = False)
    Obj.merge_volume_rgb()
    Obj.save_volume(destination, fstr = 'rgb')
    
    #print('Processing file', idx+1, 'of', len(filenameLF_LIST))













