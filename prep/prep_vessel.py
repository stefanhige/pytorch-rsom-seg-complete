#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 13:58:44 2019

@author: sgerl
"""

from pathlib import Path

import os

from classes import RSOM_vessel

from utils.get_unique_filepath import get_unique_filepath



# define folder


origin = '/home/stefan/Documents/RSOM/Diabetes/allmat'
# origin_layer = '/home/stefan/PYTHON/HQDatasetVesselAnnot/input_for_layerseg/manual_z_values'
#origin_layer = '/home/stefan/Documents/RSOM/Diabetes/selection1/prediction'
origin_layer = '/home/stefan/Documents/RSOM/Diabetes/rednoise_labels/epidermis_cutoff'
# origin = '/media/nas_ads_mwn/AG-Ntziachristos/RSOM_Data/RSOM_Diabetes/Stefan/allmat'
# origin = '/media/nas_ads_mwn/AG-Ntziachristos/RSOM_Data/RSOM_Diabetes/Stefan/'


# destination = '/media/nas_ads_mwn/AG-Ntziachristos/RSOM_Data/RSOM_Diabetes/Stefan/'
# destination = '/home/sgerl/Documents/PYTHON/TestDataset20190411/selection/other_preproccessing_tests/sliding_mip_6'
destination = '/home/stefan/Documents/RSOM/Diabetes/rednoise_labels/out_from_prep'


# mode
mode = 'list'

if mode=='dir':
    cwd = os.getcwd()
    # change directory to origin, and get a list of all files
    os.chdir(origin)
    all_files = os.listdir()
    os.chdir(cwd)
elif mode=='list':
    patterns = ['R_20170724162958_PAT004_RL01',
                'R_20170807153617_PAT014_RL01',
                'R_20170807154050_PAT014_RL02', 
                'R_20181124171923_VOL021_LL01']
    all_files = [os.path.basename(get_unique_filepath(origin, pat)[0]) for pat in patterns]

# extract the LF.mat files,
filenameLF_LIST = [el for el in all_files if el[-6:] == 'LF.mat']


for idx, filenameLF in enumerate(filenameLF_LIST):
    
    #if idx >= 1:
    #    break
    # the other ones will be automatically defined
    filenameHF = filenameLF.replace('LF.mat','HF.mat')
    
    # extract datetime
    idx_1 = filenameLF.find('_')
    idx_2 = filenameLF.find('_', idx_1+1)
    filenameSurf = 'Surf' + filenameLF[idx_1:idx_2+1] + '.mat'
    
    
    # merge paths
    fullpathHF = (Path(origin) / filenameHF).resolve()
    fullpathLF = (Path(origin) / filenameLF).resolve()
    fullpathSurf = (Path(origin) / filenameSurf).resolve()
    
    Obj = RSOM_vessel(fullpathLF, fullpathHF, fullpathSurf)
    
    Obj.readMATLAB()
    
    Obj.flatSURFACE()
    Obj.cutDEPTH()
    
    # surface for quick check
    # Obj.saveSURFACE((destination + ''), fstr = 'surf')
    
    # MIP image for quick check
    # Obj.calcMIP(do_plot = False)
    # Obj.saveMIP(destination, fstr = 'mip')
    
    # MIP 3D for annotation
    # Obj.calcMIP3D(do_plot = False)
    #Obj.saveMIP3D(destination, fstr = 'mip3d')
    
    # cut epidermis
    Obj.cutLAYER(origin_layer, mode='manual', fstr='manual_cutoff')
    
    # VOLUME
    Obj.normINTENSITY()
    Obj.rescaleINTENSITY()
    
    # debug = Obj.thresholdSEGMENTATION()
    # Obj.mathMORPH()
    
    # Obj.saveSEGMENTATION(destination, fstr='l')
    #Obj.backgroundAnnot_replaceVessel(origin_layer, 
                                      # mode='manual',
                                      # fstr='ves_cutoff')
    
    Obj.mergeVOLUME_RGB()
    Obj.saveVOLUME(destination, fstr = 'v_rgb')
    
    print('Processing file', idx+1, 'of', len(filenameLF_LIST))
