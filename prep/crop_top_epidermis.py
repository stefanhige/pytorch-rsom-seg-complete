import os
import numpy as np
from scipy import ndimage

from pathlib import Path
from classes import RSOM


def project_1d(V):
    proj = np.sum(V, axis=(1, 2))
    proj *= 1 / proj.max()
    return proj


def iterate_max(proj):
    max_idx = np.argmax(proj)
    max_val = proj[max_idx]
    indices = np.where(proj[:max_idx] < 0.1 * max_val)
    if np.any(np.ediff1d(indices) != 1):
        return iterate_max(proj[:indices[0][-1]])
    else:
        return indices[0][-1]


def main(input_dir, output_dir):
    # define folder
    all_files = os.listdir(input_dir)

    # extract the LF.mat files,
    filenameLF_LIST = [el for el in all_files if el.endswith('LF.mat') and not el.endswith('Crop_LF.mat')]

    if not filenameLF_LIST:
        print("Didn't find any files in ", input_dir)

    for idx, filenameLF in enumerate(filenameLF_LIST):
        print('Processing file', idx + 1, 'of', len(filenameLF_LIST))

        # the other ones will be automatically defined
        filenameHF = filenameLF.replace('LF.mat', 'HF.mat')

        # extract datetime
        idx_1 = filenameLF.find('_')
        idx_2 = filenameLF.find('_', idx_1 + 1)
        filenameSurf = 'Surf' + filenameLF[idx_1:idx_2 + 1] + '.mat'

        # merge paths
        fullpathHF = (Path(input_dir) / filenameHF).resolve()
        fullpathLF = (Path(input_dir) / filenameLF).resolve()
        fullpathSurf = (Path(input_dir) / filenameSurf).resolve()

        Obj = RSOM(fullpathLF, fullpathHF, fullpathSurf)

        Obj.read_matlab()
        Obj.norm_intensity()
        Obj.rescale_intensity()

        proj = project_1d(Obj.Vh_1) + project_1d(Obj.Vl_1)
        proj = ndimage.gaussian_filter1d(proj, 10, mode='constant', cval=0.0)

        cut_idx = iterate_max(proj)
        cut_idx = max(0, cut_idx-50)
        print("Cutting from 0 -", cut_idx)

        Obj.matfileLF['R'] = np.ascontiguousarray(Obj.Vl[cut_idx:, ...])
        Obj.matfileHF['R'] = np.ascontiguousarray(Obj.Vh[cut_idx:, ...])

        Obj.save_matlab(output_dir, fstr='Crop')


if __name__ == '__main__':

    # directory of MATLAB data
    input_dir = "/home/stefan/Downloads/pat01"

    output_dir = "/home/stefan/Downloads/pat01/crop"

    main(input_dir, output_dir)
