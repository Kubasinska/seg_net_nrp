import torch
from src.model import PredNet
from src.utils import train_fn, valid_fn, select_scheduler
from src.dataset_fn_nrp import get_nrp_dataloaders
from src.dataset_fn_mots import get_mots_dataloaders
from src.dataset_fn_handover import get_handover_dataloaders
from src.dataset_fn_bmw3 import get_bmw_dataloaders
from src.dataset_fn_multi2 import get_multi_dataloaders

def main():

    # General parameters
    load_model = True
    n_epochs_run = 20
    n_epochs_save = 5
    epoch_to_load = None  # None to load the last available epoch
    remove_ground = True

    # Dataset parameters
    data_params = {
        'batch_size_train': 4,
        'batch_size_valid': 16,
        'n_frames': 50,
        'tr_ratio': 0.8,
        'remove_ground': remove_ground,
        'augmentation': True,
        'dataset_dir': 'multi_shelf',  # 'mots', 'nrp', 'handover', 'multi_shelf', 'multi_small', 'bmw'
        'dataset_path': {
            'mots': r'D:\DL\datasets\kitti\mots',
            'nrp': r'D:\DL\datasets\nrp\training_room',
            'handover': r'D:\DL\datasets\nrp\handover',
            'multi_shelf': r'/workspace/dataset/multi_shelf',
            'multi_small': r'D:\DL\datasets\nrp\multi_small',
            'bmw': r'D:\DL\datasets\nrp\bmw'}}
    dataloader_fn = {'mots': get_mots_dataloaders,  # TODO: either code them or make only one file loader
                     'nrp': get_nrp_dataloaders,
                     'handover': get_handover_dataloaders,
                     'multi_shelf': get_multi_dataloaders,
                     'multi_small': get_multi_dataloaders,
                     'bmw': get_bmw_dataloaders}[data_params['dataset_dir']]
    train_dl, valid_dl, n_classes = dataloader_fn(**data_params)

    # Model parameters
    n_layers = 3
    model_params = {
        'n_layers': n_layers,
        'do_time_aligned': True,
        'n_classes': n_classes,
        'td_layers': ('H', 'H', 'H', 'H', 'H', 'H')[:n_layers],  # 'Hgru', 'Illusory', 'Lstm', 'Conv'
        'img_layers': tuple([l for l in [0, 2] if l < n_layers]),  # set as [] for not using it
        'seg_layers': tuple([l for l in [0, 1] if l < n_layers]),  # set as [] for not using it
        'bu_channels': (64, 128, 256, 512, 1024)[:n_layers],
        'td_channels': (64, 128, 256, 512, 1024)[:n_layers],
        'device': 'cuda'}  # 'cuda', 'cpu'

    # Loss parameters
    loss_params = {
        'n_backprop_frames': 10,
        't_start': 1,
        'remove_ground': remove_ground,
        'latent': (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)[:n_layers],
        'prd_mse': 0.0 if len(model_params['img_layers']) > 0 else 0.0,
        'prd_mae': 1.0 if len(model_params['img_layers']) > 0 else 0.0,
        'seg_dic': 1.0 if len(model_params['seg_layers']) > 0 else 0.0,
        'seg_foc': 1.0 if len(model_params['seg_layers']) > 0 else 0.0}

    # Training parameters
    lr = 5e-4
    lr_params = {
        'scheduler_type': 'multistep',  # 'multistep', 'cosine', 'onecycle'
        'optimizer': {'lr': lr, 'betas': (0.9, 0.98), 'eps': 1e-8},
        'multistep': {'milestones': range(5, 10000, 5), 'gamma': 0.75},
        'cosine': {'first_cycle_steps': 10, 'cycle_mult': 1.0, 'max_lr': lr,
                'min_lr': lr / 100, 'warmup_steps': 2, 'gamma': 1.0},
        'onecycle': {'max_lr': lr, 'steps_per_epoch': len(train_dl), 'epochs': n_epochs_run}}

    # Load the model, optimizer and scheduler
    model_name = 'BU' + str(model_params['bu_channels']) + '_TD' + str(model_params['td_channels'])\
               + '_TL' + str(model_params['td_layers']) + '_IL' + str(model_params['img_layers'])\
               + '_PL' + str(model_params['seg_layers'])
    model_name = model_name.replace('.', '-').replace(',', '-').replace(' ', '').replace("'", '')
    model_name = 'Michael_ckpt'
    if not load_model:
        print(f'\nCreating model: {model_name}')
        model = PredNet(model_name, **model_params)
        train_losses, valid_losses = [], []
        optimizer = torch.optim.AdamW(model.parameters(), **lr_params['optimizer'])
        scheduler = select_scheduler(optimizer, lr_params)
    else:
        print(f'\nLoading model: {model_name}')
        model, optimizer, scheduler, train_losses, valid_losses = PredNet.load_model(model_name,
                                                                                     epoch_to_load,
                                                                                     lr_params)

    # Train the network
    last_epoch = len(valid_losses) // (1 + len(valid_dl) // data_params['batch_size_valid'])
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'Training network ({n_params} trainable parameters)')
    for epoch in range(last_epoch, last_epoch + n_epochs_run):
        print(f'\nRunning epoch {epoch}:')

        model.save_model(optimizer, scheduler, train_losses, valid_losses, epoch, n_epochs_save)
        exit()

        train_losses.extend(train_fn(train_dl, model, optimizer, scheduler, loss_params, epoch))
        valid_losses.extend(valid_fn(valid_dl, model, loss_params, epoch))
        model.save_model(optimizer, scheduler, train_losses, valid_losses, epoch, n_epochs_save)

if __name__ == '__main__':
    main()


# import torch
# import os
# from cosine_annealing_warmup import CosineAnnealingWarmupRestarts
# from src.model import PredNet
# from src.utils import train_fn, valid_fn
# from src.dataset_fn_nrp import get_nrp_dataloaders
# from src.dataset_fn_mots import get_mots_dataloaders, get_sqm_dataloaders
# from src.dataset_fn_handover import get_handover_dataloaders
# # from src.dataset_fn_bmw import get_bmw_dataloaders
# # from src.dataset_fn_bmw2 import get_bmw_dataloaders
# from src.dataset_fn_bmw3 import get_bmw_dataloaders
# # from src.dataset_fn_multi import get_multi_dataloaders
# from src.dataset_fn_multi2 import get_multi_dataloaders

# # Model parameters
# load_model, do_time_aligned, n_layers = False, True, 3
# do_untouched_bu = False
# batch_size_train, batch_size_valid = 4, 16
# prd_layers = tuple([l for l in [0] if l < n_layers])
# seg_layers = tuple([l for l in [1, 2] if l < n_layers])
# bu_channels = (64, 128, 256, 512)[:n_layers]
# td_channels = (64, 128, 256, 512)[:n_layers]
# # td_layers = ('C', 'C', 'C', 'C')[:n_layers]  # 'Hgru', 'Illusory', 'Lstm', 'Conv'
# td_layers = ('H', 'H', 'H', 'H')[:n_layers]  # 'Hgru', 'Illusory', 'Lstm', 'Conv'
# dropout_rates = (0.0, 0.0, 0.0, 0.0)[:n_layers]
# device = 'cuda'  # 'cuda', 'cpu'

# # Training parameters
# n_epochs_run, n_epochs_save, epoch_to_load = 1000, 5, None
# learning_rate, lr_decay_time, lr_decay_rate, betas = 1e-4, 50, 0.75, (0.9, 0.98)
# first_cycle_steps, cycle_mult, max_lr, min_lr, warmup_steps, gamma = 10, 1.0, 1e-4, 1e-5, 2, 1.0
# scheduler_type = 'multistep'  # 'multistep', 'cosannealwarmuprestart'
# loss_w = {
#     'latent': (5.0, 0.0, 0.0, 0.0)[:n_layers],  # now this is image loss
#     'prd_mae': 3.0 if len(prd_layers) > 0 else 0.0,  # put to zero?
#     'prd_mse': 0.0 if len(prd_layers) > 0 else 0.0,
#     'prd_bce': 0.0 if len(prd_layers) > 0 else 0.0,
#     'seg_mae': 0.0 if len(seg_layers) > 0 else 0.0,
#     'seg_mse': 0.0 if len(seg_layers) > 0 else 0.0,
#     'seg_bce': 0.0 if len(seg_layers) > 0 else 0.0,
#     'seg_foc': 1.0 if len(seg_layers) > 0 else 0.0,
#     'seg_dic': 1.0 if len(seg_layers) > 0 else 0.0}
# do_prediction = not (len(prd_layers) == 0 or \
#     sum([loss_w['prd_' + k] for k in ['mae', 'mse', 'bce']]) == 0)
# do_segmentation = not (len(seg_layers) == 0 or \
#     sum([loss_w['seg_' + k] for k in ['mae', 'mse', 'bce', 'foc', 'dic']]) == 0)

# # Build model name
# model_name = \
#       f'TA{int(do_time_aligned)}_BU{bu_channels}_TD{td_channels}_TL{td_layers}'\
#       f'_PL{prd_layers}_SL{seg_layers}_DR{tuple([int(10 * r) for r in dropout_rates])}'
# model_name = model_name.replace('.', '-').replace(',', '-').replace(' ', '').replace("'", '')

# # Dataset
# dataset_dir = r'D:\DL\datasets\nrp'
# dataset_type = 'multi_shelf'  # 'mots', 'mots_sqm', 'nrp', 'handover', 'multi_shelf', 'multi_small', 'bmw', 'ovis'
# n_frames, n_backprop_frames, t_start = 59, 5, n_layers
# augmentation, remove_ground, tr_ratio = True, True, 0.8

# if dataset_type == 'nrp':
#     # dataset_path = os.path.join(dataset_dir, 'training_room', 'training_room_dataset_00.h5')  # 1000 x 128 x 128 x 3
#     dataset_path = os.path.join(dataset_dir, 'training_room', 'training_room_dataset_01.h5')  # 300 x 320 x 320 x 3
#     n_samples, n_classes = 300, 3 + int(not remove_ground)
#     train_dl, valid_dl = get_nrp_dataloaders(dataset_path, tr_ratio, n_samples, 
#                                              batch_size_train, batch_size_valid,
#                                              n_classes, augmentation=augmentation,
#                                              remove_ground=remove_ground, speedup_factor=1)

# elif dataset_type in ['handover', 'multi_shelf', 'multi_small', 'bmw']:
#     dataset_path = os.path.join(dataset_dir, dataset_type)
#     dataloader_fn_dict = {'handover': get_handover_dataloaders,
#                           'multi_small': get_multi_dataloaders,
#                           'multi_shelf': get_multi_dataloaders,
#                           'bmw': get_bmw_dataloaders}
#     dataloader_fn = dataloader_fn_dict[dataset_type]
#     train_dl, valid_dl, n_classes = dataloader_fn(dataset_path, tr_ratio,
#                                                   batch_size_train, batch_size_valid, n_frames,
#                                                   augmentation=augmentation, remove_ground=remove_ground)
                                                  
# # elif 'mots' in dataset_type:
# #     dataset_path = r'D:\DL\datasets\kitti\mots'
# #     sqm_dataset_path = r'D:\DL\datasets\kitti\mots\sqm'
# #     n_classes = 3 + int(not remove_ground)
# #     train_dl, valid_dl = get_mots_dataloaders(dataset_path, tr_ratio, batch_size_train,
# #                                               batch_size_valid, n_frames, augmentation=augmentation,
# #                                               n_classes=n_classes, remove_ground=remove_ground)
# #     if 'sqm' in dataset_type:
# #         sqm_losses = []
# #         sqm_dl = get_sqm_dataloaders(sqm_dataset_path, n_classes, remove_ground)
# # 
# # elif dataset_type == 'ovis':
# #     dataset_path = r'D:\DL\datasets\ovis'
# #     train_dl, valid_dl, n_classes = get_ovis_dataloaders(dataset_path, tr_ratio,
# #                                                          batch_size_train, batch_size_valid, n_frames,
# #                                                          augmentation=augmentation, remove_ground=remove_ground)

# # Load the model
# if not load_model:
#     print(f'\nCreating model: {model_name}')
#     model = PredNet(model_name,
#                     n_classes,
#                     n_layers,
#                     td_layers,
#                     seg_layers,
#                     bu_channels,
#                     td_channels,
#                     do_segmentation,
#                     device)
#     train_losses, valid_losses, last_epoch = [], [], 0
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=betas)
#     if scheduler_type == 'multistep':
#         milestones=range(lr_decay_time,10 * (n_epochs_run + 1), lr_decay_time)
#         scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
#                                                          milestones,
#                                                          lr_decay_rate)
#     elif scheduler_type == 'cosannealwarmuprestart':
#         scheduler = CosineAnnealingWarmupRestarts(optimizer,
#                                                   first_cycle_steps=first_cycle_steps,
#                                                   cycle_mult=cycle_mult,
#                                                   max_lr=max_lr,
#                                                   min_lr=min_lr,
#                                                   warmup_steps=warmup_steps,
#                                                   gamma=gamma)
#     train_losses, valid_losses = [], []
# else:
#     print(f'\nLoading model: {model_name}')
#     lr_params = [scheduler_type, learning_rate, lr_decay_time, lr_decay_rate, betas,
#                  first_cycle_steps, cycle_mult, max_lr, min_lr, warmup_steps, gamma, betas]
#     model, optimizer, scheduler, train_losses, valid_losses = PredNet.load_model(model_name,
#                                                                                  n_epochs_run,
#                                                                                  epoch_to_load,
#                                                                                  lr_params)
#     last_epoch = scheduler.last_epoch

# # Train the network
# n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
# print(f'Training network ({n_params} trainable parameters)')
# for epoch in range(last_epoch, last_epoch + n_epochs_run):
#     print(f'\nRunning epoch {epoch}:')
#     train_losses.extend(train_fn(train_dl, model, optimizer, loss_w, remove_ground,
#                                  t_start, n_backprop_frames, epoch))
#     valid_losses.extend(valid_fn(valid_dl, model, loss_w, remove_ground, t_start, epoch))
#     # if 'sqm' in dataset_type:
#     #     sqm_losses.append(valid_fn(sqm_dl, model, loss_w, t_start, epoch))
#     scheduler.step()
#     if (epoch + 1) % n_epochs_save == 0:
#         model.save_model(optimizer, scheduler, train_losses, valid_losses)
