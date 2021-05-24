
import torch
import os
from laynet._metrics import custom_loss_1_smooth, bce_and_smooth
from laynet import LayerNet, LayerNetBase

# torch.backends.cudnn.benchmark = True
mode = 'train'

if mode == 'train':

    sdesc = ['test']

    root_dir = '/home/stefan/RSOM/testing/onefile'

    DEBUG = False
    #DEBUG = True

    out_dir = '/home/stefan/RSOM/testing/output'

    model_type = 'unet'

    #os.environ["CUDA_VISIBLE_DEVICES"]='4'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for idx in range(len(sdesc)):
        root_dir = root_dir
        train_dir = root_dir
        eval_dir = root_dir

        #train_dir = os.path.join(root_dir, 'train')
        #eval_dir = os.path.join(root_dir, 'val')
        pred_dir = train_dir

        dirs={'train':train_dir,'eval':eval_dir, 'model':'', 'pred':pred_dir, 'out': out_dir}

        net1 = LayerNet(device=device,
                        sdesc=sdesc[idx],
                        model_depth=1,
                        model_type=model_type,
                        dataset_zshift=(-50, 200),
                        dirs=dirs,
                        optimizer='Adam',
                        initial_lr=1e-4,
                        scheduler_patience=3,
                        lossfn=torch.nn.BCEWithLogitsLoss(reduction='sum'),
                        lossfn_smoothness=0,
                        lossfn_window=5,
                        lossfn_spatial_weight_scale=False,
                        epochs=2,
                        dropout=True,
                        class_weight=None,
                        DEBUG=DEBUG,
                        batch_size=2,
                        decision_boundary=0.5
                        )

        net1.printConfiguration()
        net1.save_code_status()
        net1.train_all_epochs()
        net1.predict()
        net1.save_model()

elif mode == 'predict':

    os.environ["CUDA_VISIBLE_DEVICES"]='4'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
   
    pred_dir = '/home/stefan/RSOM/testing/onefile'

    # model_dir ='/home/gerlstefan/models/layerseg/test/mod_191101_depth5.pt'
    model_dir ='/home/stefan/RSOM/testing/output/210524-03-test/mod210524-03_best_.pt'

    out_dir ='/home/stefan/RSOM/testing/output'
    model_type = 'unet'
    
    net1 = LayerNetBase(
            dirs={'model': model_dir,
                  'pred': pred_dir,
                  'out': out_dir},
            device=device,
            model_depth=1,
            batch_size=2,
            model_type=model_type)
    net1.predict()





