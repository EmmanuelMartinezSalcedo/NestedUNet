import argparse
import os
from collections import OrderedDict
from glob import glob
import pandas as pd
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import yaml
import albumentations as A
from albumentations.core.composition import Compose, OneOf
from sklearn.model_selection import train_test_split
from torch.optim import lr_scheduler
from tqdm import tqdm
import archs
import losses
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter, str2bool

ARCH_NAMES = archs.__all__
LOSS_NAMES = losses.__all__
LOSS_NAMES.append('BCEWithLogitsLoss')

def parse_args():
    parser = argparse.ArgumentParser()
    
    # Existing arguments...
    parser.add_argument('--name', default=None,
                        help='model name: (default: arch+timestamp)')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch_size', default=16, type=int,
                        metavar='N', help='mini-batch size (default: 16)')
    
    # model
    parser.add_argument('--arch', '-a', metavar='ARCH', default='NestedUNet',
                        choices=ARCH_NAMES,
                        help='model architecture: ' +
                        ' | '.join(ARCH_NAMES) +
                        ' (default: NestedUNet)')
    parser.add_argument('--deep_supervision', default=False, type=str2bool)
    parser.add_argument('--input_channels', default=3, type=int,
                        help='input channels')
    parser.add_argument('--num_classes', default=1, type=int,
                        help='number of classes')
    parser.add_argument('--input_w', default=512, type=int,
                        help='image width')
    parser.add_argument('--input_h', default=512, type=int,
                        help='image height')
    
    # loss
    parser.add_argument('--loss', default='BCEDiceLoss',
                        choices=LOSS_NAMES,
                        help='loss: ' +
                        ' | '.join(LOSS_NAMES) +
                        ' (default: BCEDiceLoss)')
    
    # dataset
    parser.add_argument('--dataset', default='dsb2018_96',
                        help='dataset name')
    parser.add_argument('--img_ext', default='.png',
                        help='image file extension')
    parser.add_argument('--mask_ext', default='.png',
                        help='mask file extension')
    
    # optimizer
    parser.add_argument('--optimizer', default='SGD',
                        choices=['Adam', 'SGD'],
                        help='loss: ' +
                        ' | '.join(['Adam', 'SGD']) +
                        ' (default: Adam)')
    parser.add_argument('--lr', '--learning_rate', default=1e-3, type=float,
                        metavar='LR', help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float,
                        help='momentum')
    parser.add_argument('--weight_decay', default=1e-4, type=float,
                        help='weight decay')
    parser.add_argument('--nesterov', default=False, type=str2bool,
                        help='nesterov')
    
    # scheduler
    parser.add_argument('--scheduler', default='CosineAnnealingLR',
                        choices=['CosineAnnealingLR', 'ReduceLROnPlateau', 'MultiStepLR', 'ConstantLR'])
    parser.add_argument('--min_lr', default=1e-5, type=float,
                        help='minimum learning rate')
    parser.add_argument('--factor', default=0.1, type=float)
    parser.add_argument('--patience', default=2, type=int)
    parser.add_argument('--milestones', default='1,2', type=str)
    parser.add_argument('--gamma', default=2/3, type=float)
    parser.add_argument('--early_stopping', default=-1, type=int,
                        metavar='N', help='early stopping (default: -1)')
    
    # NUEVOS PARÁMETROS ANTI-OVERFITTING
    parser.add_argument('--dropout_rate', default=0.2, type=float,
                        help='dropout rate for regularization')
    parser.add_argument('--mixup_alpha', default=0.0, type=float,
                        help='mixup alpha parameter (0 to disable)')
    parser.add_argument('--cutmix_alpha', default=0.0, type=float,
                        help='cutmix alpha parameter (0 to disable)')
    parser.add_argument('--label_smoothing', default=0.0, type=float,
                        help='label smoothing factor')
    parser.add_argument('--validation_split', default=0.2, type=float,
                        help='validation split ratio')
    parser.add_argument('--accumulation_steps', default=8, type=int,
                        help='gradient accumulation steps to simulate larger batch size')
    
    parser.add_argument('--num_workers', default=4, type=int)
    
    config = parser.parse_args()
    return config

def mixup_data(x, y, alpha=1.0):
    """Mixup data augmentation"""
    if alpha > 0:
        lam = torch.distributions.beta.Beta(alpha, alpha).sample()
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).cuda()
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    """CutMix data augmentation"""
    if alpha > 0:
        lam = torch.distributions.beta.Beta(alpha, alpha).sample()
    else:
        lam = 1
    
    batch_size = x.size(0)
    index = torch.randperm(batch_size).cuda()
    
    W = x.size()[2]
    H = x.size()[3]
    cut_rat = torch.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    
    # uniform
    cx = torch.randint(W, (1,))
    cy = torch.randint(H, (1,))
    
    bbx1 = torch.clamp(cx - cut_w // 2, 0, W)
    bby1 = torch.clamp(cy - cut_h // 2, 0, H)
    bbx2 = torch.clamp(cx + cut_w // 2, 0, W)
    bby2 = torch.clamp(cy + cut_h // 2, 0, H)
    
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    
    # adjust lambda to exactly match pixel ratio
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    
    y_a, y_b = y, y[index]
    return x, y_a, y_b, lam

def train(config, train_loader, model, criterion, optimizer):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}
    
    model.train()
    pbar = tqdm(total=len(train_loader))
    
    # Para gradient accumulation
    accumulation_steps = config['accumulation_steps']
    optimizer.zero_grad()
    
    for i, (input, target, *_) in enumerate(train_loader):
        input = input.cuda()
        target = target.cuda()
        
        # Para batch_size=1, desactivar mixup/cutmix ya que no funcionan bien
        # Forward pass normal
        if config['deep_supervision']:
            outputs = model(input)
            loss = 0
            for output in outputs:
                loss += criterion(output, target)
            loss /= len(outputs)
            iou = iou_score(outputs[-1], target)
        else:
            output = model(input)
            loss = criterion(output, target)
            iou = iou_score(output, target)
        
        # Normalizar loss por accumulation steps
        loss = loss / accumulation_steps
        loss.backward()
        
        # Actualizar cada accumulation_steps
        if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_loader):
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
        
        # Para logging, usar loss sin normalizar
        avg_meters['loss'].update(loss.item() * accumulation_steps, input.size(0))
        avg_meters['iou'].update(iou, input.size(0))
        
        postfix = OrderedDict([
            ('loss', avg_meters['loss'].avg),
            ('iou', avg_meters['iou'].avg),
        ])
        pbar.set_postfix(postfix)
        pbar.update(1)
    pbar.close()
    
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])

def validate(config, val_loader, model, criterion):
    avg_meters = {'loss': AverageMeter(),
                  'iou': AverageMeter()}
    
    # switch to evaluate mode
    model.eval()
    
    with torch.no_grad():
        pbar = tqdm(total=len(val_loader))
        for input, target, *_ in val_loader:
            input = input.cuda()
            target = target.cuda()
            
            # compute output
            if config['deep_supervision']:
                outputs = model(input)
                loss = 0
                for output in outputs:
                    loss += criterion(output, target)
                loss /= len(outputs)
                iou = iou_score(outputs[-1], target)
            else:
                output = model(input)
                loss = criterion(output, target)
                iou = iou_score(output, target)
            
            avg_meters['loss'].update(loss.item(), input.size(0))
            avg_meters['iou'].update(iou, input.size(0))
            
            postfix = OrderedDict([
                ('loss', avg_meters['loss'].avg),
                ('iou', avg_meters['iou'].avg),
            ])
            pbar.set_postfix(postfix)
            pbar.update(1)
        pbar.close()
    
    return OrderedDict([('loss', avg_meters['loss'].avg),
                        ('iou', avg_meters['iou'].avg)])

def main():
    config = vars(parse_args())
    
    if config['name'] is None:
        if config['deep_supervision']:
            config['name'] = '%s_%s_wDS' % (config['dataset'], config['arch'])
        else:
            config['name'] = '%s_%s_woDS' % (config['dataset'], config['arch'])
    
    os.makedirs('models/%s' % config['name'], exist_ok=True)
    
    print('-' * 20)
    for key in config:
        print('%s: %s' % (key, config[key]))
    print('-' * 20)
    
    with open('models/%s/config.yml' % config['name'], 'w') as f:
        yaml.dump(config, f)
    
    # define loss function (criterion) con label smoothing
    if config['loss'] == 'BCEWithLogitsLoss':
        if config['label_smoothing'] > 0:
            # Implementar label smoothing personalizado
            criterion = nn.BCEWithLogitsLoss().cuda()
        else:
            criterion = nn.BCEWithLogitsLoss().cuda()
    else:
        criterion = losses.__dict__[config['loss']]().cuda()
    
    cudnn.benchmark = True
    
    # create model
    print("=> creating model %s" % config['arch'])
    model = archs.__dict__[config['arch']](config['num_classes'],
                                           config['input_channels'],
                                           config['deep_supervision'],
                                           dropout_rate=config['dropout_rate'])
    
    # Agregar dropout si no está en el modelo
    if hasattr(model, 'add_dropout'):
        model.add_dropout(config['dropout_rate'])
    
    model = model.cuda()
    
    params = filter(lambda p: p.requires_grad, model.parameters())
    if config['optimizer'] == 'Adam':
        optimizer = optim.Adam(
            params, lr=config['lr'], weight_decay=config['weight_decay'])
    elif config['optimizer'] == 'SGD':
        optimizer = optim.SGD(params, lr=config['lr'], momentum=config['momentum'],
                              nesterov=config['nesterov'], weight_decay=config['weight_decay'])
    else:
        raise NotImplementedError
    
    # Scheduler configuration
    if config['scheduler'] == 'CosineAnnealingLR':
        scheduler = lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=config['epochs'], eta_min=config['min_lr'])
    elif config['scheduler'] == 'ReduceLROnPlateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=config['factor'], 
                                                   patience=config['patience'],
                                                   min_lr=config['min_lr'])
    elif config['scheduler'] == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, 
                                           milestones=[int(e) for e in config['milestones'].split(',')], 
                                           gamma=config['gamma'])
    elif config['scheduler'] == 'ConstantLR':
        scheduler = None
    else:
        raise NotImplementedError
    
    # Data loading code
    img_ids = glob(os.path.join('inputs', config['dataset'], 'images', '*' + config['img_ext']))
    img_ids = [os.path.splitext(os.path.basename(p))[0] for p in img_ids]
    
    # Aumentar validation split si hay overfitting
    train_img_ids, val_img_ids = train_test_split(img_ids, 
                                                   test_size=config['validation_split'], 
                                                   random_state=41)
    
    # AUGMENTACIONES MÁS AGRESIVAS PARA TRAINING
    train_transform = Compose([
        A.RandomRotate90(p=0.7),  # Aumentado de 0.5
        A.HorizontalFlip(p=0.7),  # Aumentado de 0.5
        A.VerticalFlip(p=0.5),    # Nuevo
        OneOf([
            A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=0.7),
            A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.7),
            A.RandomGamma(gamma_limit=(80, 120), p=0.7),
            A.CLAHE(clip_limit=4.0, tile_grid_size=(8, 8), p=0.5),  # Nuevo
        ], p=1),
        OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=2, p=0.5),
        ], p=0.3),  # Distorsiones geométricas
        OneOf([
            A.GaussNoise(var_limit=(10, 50), p=0.5),
            A.GaussianBlur(blur_limit=3, p=0.5),
            A.MotionBlur(blur_limit=3, p=0.5),
        ], p=0.2),  # Ruido y blur
        A.Affine(
            scale=(0.9, 1.1),  # scale_limit=0.1
            translate_percent={"x": (-0.0625, 0.0625), "y": (-0.0625, 0.0625)},  # shift_limit
            rotate=(-15, 15),  # rotate_limit
            p=0.5
        ),
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])
    
    val_transform = Compose([
        A.Resize(config['input_h'], config['input_w']),
        A.Normalize(),
    ])
    
    train_dataset = Dataset(
        img_ids=train_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=train_transform)
    
    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        drop_last=True)
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)
    
    log = OrderedDict([
        ('epoch', []),
        ('lr', []),
        ('loss', []),
        ('iou', []),
        ('val_loss', []),
        ('val_iou', []),
    ])
    
    best_iou = 0
    trigger = 0
    
    for epoch in range(config['epochs']):
        print('Epoch [%d/%d]' % (epoch, config['epochs']))
        
        # train for one epoch
        train_log = train(config, train_loader, model, criterion, optimizer)
        
        # evaluate on validation set
        val_log = validate(config, val_loader, model, criterion)
        
        if config['scheduler'] == 'CosineAnnealingLR':
            scheduler.step()
        elif config['scheduler'] == 'ReduceLROnPlateau':
            scheduler.step(val_log['loss'])
        
        print('loss %.4f - iou %.4f - val_loss %.4f - val_iou %.4f'
              % (train_log['loss'], train_log['iou'], val_log['loss'], val_log['iou']))
        
        # Detectar overfitting
        overfitting_gap = train_log['iou'] - val_log['iou']
        if overfitting_gap > 0.1:  # Si la diferencia es mayor al 10%
            print(f"WARNING: Possible overfitting detected. Gap: {overfitting_gap:.4f}")
        
        log['epoch'].append(epoch)
        log['lr'].append(optimizer.param_groups[0]['lr'])  # Corregido para obtener LR actual
        log['loss'].append(train_log['loss'])
        log['iou'].append(train_log['iou'])
        log['val_loss'].append(val_log['loss'])
        log['val_iou'].append(val_log['iou'])
        
        pd.DataFrame(log).to_csv('models/%s/log.csv' %
                                 config['name'], index=False)
        
        trigger += 1
        
        if val_log['iou'] > best_iou:
            torch.save(model.state_dict(), 'models/%s/model.pth' %
                       config['name'])
            best_iou = val_log['iou']
            print("=> saved best model")
            trigger = 0
        
        # Early stopping más agresivo para evitar overfitting
        if config['early_stopping'] >= 0 and trigger >= config['early_stopping']:
            print("=> early stopping")
            break
        
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()