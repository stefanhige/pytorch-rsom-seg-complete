
import torch
import os
from laynet._metrics import custom_loss_1_smooth, bce_and_smooth, LossScheduler
from laynet import LayerNet, LayerNetBase

import types
# torch.backends.cudnn.benchmark = True
mode = 'train'

if mode == 'train':
    
    sdesc = []
    sliding_window_size = []
    class_weights = []
    
    i = 5
    for i in [1, 5, 9]:
        sdesc.append("50s_slid_mip" + str(i) + '_20ep')
        sliding_window_size.append(i)




    # root_dir = '/home/stefan/RSOM/testing/onefile'
    # root_dir = '/home/gerlstefan/data/layerunet/dataloader_dev'
    # root_dir = '/home/gerlstefan/data/layerunet/fullDatasetExtended/labeled'
    root_dir = '/home/gerlstefan/data/layerunet/fullDatasetExtended/labeled_fixed'
    # root_dir = '/home/gerlstefan/data/layerunet/fullDatasetExtended/labeled_reduced_intensity'
    # root_dir = '/home/gerlstefan/data/layerunet/fullDataset/labeled'
    DEBUG = False
    # DEBUG = True

    out_dir = '/home/gerlstefan/data/layerunet/output'

    model_type = 'unet'

    os.environ["CUDA_VISIBLE_DEVICES"]='7'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    for idx in range(len(sdesc)):
        # train_dir = root_dir
        # eval_dir = root_dir

        train_dir = os.path.join(root_dir, 'train')
        eval_dir = os.path.join(root_dir, 'val')
        pred_dir = eval_dir

        dirs={'train': train_dir,
              'eval': eval_dir,
              'model':'',
              'pred': pred_dir,
              'out': out_dir}

        aug_params = types.SimpleNamespace()
        aug_params.zshift = (-75, 100)
        aug_params.sliding_window_size = sliding_window_size[idx]

        # loss_scheduler = LossScheduler(base_loss=torch.nn.BCEWithLogitsLoss(reduction='sum'),
        #                                additional_loss=torch.nn.BCEWithLogitsLoss(reduction='sum'),
        #                                epoch_start=1,
        #                                factor=1.5,
                                       # n_epochs=2)


        net1 = LayerNet(device=device,
                        sdesc=sdesc[idx],
                        model_depth=5,
                        model_type=model_type,
                        aug_params=aug_params,
                        dirs=dirs,
                        optimizer='Adam',
                        initial_lr=1e-4,
                        scheduler_patience=5,
                        lossfn=torch.nn.BCEWithLogitsLoss(reduction='sum'),
                            # pos_weight=torch.Tensor([class_weights[idx]]).to(device)),
                        # loss_scheduler=loss_scheduler,
                        epochs=20,
                        dropout=True,
                        DEBUG=DEBUG,
                        batch_size=6,
                        decision_boundary=0.5
                        )

        net1.printConfiguration()
        net1.save_code_status()
        net1.train_all_epochs()
        net1.predict()
        net1.save_model()
        net1.calc_metrics()
        net1.metricCalculator.plot_dice(os.path.join(net1.dirs['out'],'thvsdice_last.png'))

        # metrics on "best model"
        net1.model.load_state_dict(net1.best_model)
        net1.printandlog("")
        net1.printandlog("=================================================")
        net1.printandlog("Metrics from 'best' model")
        net1.out_pred_dir = net1.out_pred_dir + '_best_on_eval'
        os.mkdir(net1.out_pred_dir)
        net1.predict(save=True)
        net1.calc_metrics()
        net1.metricCalculator.plot_dice(os.path.join(net1.dirs['out'],'thvsdice_best.png'))

        # break

elif mode == 'predict':

    os.environ["CUDA_VISIBLE_DEVICES"]='7'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
   
    pred_dir = '/home/gerlstefan/data/layerunet/output/210602-01-predict-new-labels/input'

    # model_dir ='/home/gerlstefan/models/layerseg/test/mod_191101_depth5.pt'
    model_dir ='/home/gerlstefan/data/layerunet/output/210531-00-_slid_mip5/mod210531-00_last_.pt'

    out_dir ='/home/gerlstefan/data/layerunet/output/210602-01-predict-new-labels'
    model_type = 'unet'
    
    net1 = LayerNetBase(
            dirs={'model': model_dir,
                  'pred': pred_dir,
                  'out': out_dir},
            device=device,
            model_depth=5,
            sliding_window_size=5,
            batch_size=6,
            model_type=model_type)
    net1.predict()

    with open(os.path.join(out_dir, 'MODEL.txt'), 'w') as fd:
        fd.write(model_dir)
