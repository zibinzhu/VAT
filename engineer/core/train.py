'''
@author:lingteng qiu
@name:training method

'''
import os
from utils.structure import AverageMeter
import time
import torch.nn as nn
import torch
import utils.optimizer as optim
import numpy as np
import logging
import torch.distributed as dist
import time
from utils.distributed import reduce_tensor,save_checkpoints
from engineer.utils.gallery import save_gallery
from engineer.core.eval import test_epoch

logger = logging.getLogger('logger.trainer')

def train_epochs(model, optimizer, cfg, args, train_data_set, train_loader, test_loader, resume_epoch, gallery_id):
    '''Training epoch method based on open mmlab
    Parameters:
        model: training model
        optimizer: optimizer for training
        cfg (CfgNode): configs. Details can be found in
            configs/PIFu_Render_People_HG.py
        args: option parameters
        train_loader: train dataloader iterator        
        test_loader: test dataloader file
        resume_epoch: resume epoch
        gallery_id: dir you save your gallery results
    Return:
        None
    '''

    best_iou = float('-inf')
    len_train_loader = len(train_loader)
    for epoch in range(resume_epoch,cfg.num_epoch):
        epoch_start_time = time.time()
        logger.info("training epoch {}".format(epoch))
        model.train()
        #define train_loss
        train_loss = AverageMeter()
        c_loss = AverageMeter()
        f_loss = AverageMeter()
        n_loss = AverageMeter()
        iter_data_time = time.time()
        for idx,data in enumerate(train_loader): 
            #adjust learning rate
            lr_epoch = epoch+idx/len_train_loader
            optim.adjust_learning_rate(optimizer,lr_epoch,cfg)

            names = data['name']
            #ray_o = data['ray_o'].cuda()
            ray_map = data['ray_map'].cuda()
            img = data['img'].cuda()
            #mask = data['mask'].cuda()
            calib = data['calib'].cuda()
            samples = data['samples'].cuda()
            labels = data['labels'].cuda()
            #normal = data['normal'].cuda()
            # if cfg.normal:
            #     img = torch.cat([img, normal], dim=1)

            iter_start_time = time.time()
            
            # preds, loss, closs, floss, nloss = model(images=img, normals=normal, ray_map=ray_map, masks=mask, points=samples, calibs=calib, labels=labels)

            preds, loss = model(images=img, ray_map=ray_map, points=samples, calibs=calib, labels=labels)
            
            """ loss = loss.sum()
            closs = closs.sum()
            floss = floss.sum()
            nloss = nloss.sum()
            """

            loss = loss.sum()
            #loss = loss / (cfg.batch_size * cfg.data.train.num_sample_points)
            """ closs = closs / (cfg.batch_size * cfg.data.train.num_sample_points)
            floss = floss / (cfg.batch_size * cfg.data.train.num_sample_points) """

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if args.dist:
                loss = reduce_tensor(loss)
            train_loss.update(loss.item()/cfg.num_gpu, cfg.batch_size)
            """ c_loss.update(closs.item()/cfg.num_gpu, cfg.batch_size)
            f_loss.update(floss.item()/cfg.num_gpu, cfg.batch_size)
            n_loss.update(nloss.item()/cfg.num_gpu, cfg.batch_size) """

            #time calculate
            iter_net_time = time.time()
            eta = int(((iter_net_time - epoch_start_time) / (idx + 1)) * len_train_loader - (
                    iter_net_time - epoch_start_time))

            #training visible
            if idx % args.freq_plot == 0:
                word_handler = 'Name: {0} | Epoch: {1} | {2}/{3} | Err: {4:.06f} | cErr: {5:.06f} | fErr: {6:.06f} | nErr: {7:.06f} | LR: {8:.06f}  | dataT: {9:.05f} | netT: {10:.05f}'.format( 
                    cfg.name,epoch,idx,len_train_loader,train_loss.avg,c_loss.avg,f_loss.avg,n_loss.avg,optim.get_lr_at_epoch(cfg,lr_epoch), 
                    iter_start_time - iter_data_time,iter_net_time - iter_start_time, 
                    int(eta//3600),int((eta%3600)//60),eta%60)
                if (not args.dist) or dist.get_rank() == 0:
                    logger.info(word_handler)
            if idx!=0 and idx % args.freq_gallery ==0:
                save_gallery(preds, samples, names, gallery_id['train'], epoch)
            iter_data_time = time.time()
            
        if epoch>0 and epoch % cfg.save_fre_epoch ==0:
            logger.info("save model: epoch {}!".format(epoch))
            save_checkpoints(model,epoch,optimizer,gallery_id['save_path'],args)  
        #test 
        if epoch>= cfg.start_val_epoch and epoch % cfg.val_epoch ==0:
            test_metric = test_epoch(model, cfg, args, test_loader, epoch ,gallery_id['test'])
            if best_iou<test_metric['iou']:
                best_iou = test_metric['iou']
                save_checkpoints(model,epoch,optimizer,gallery_id['save_path'],args,best=True)
                
        # if epoch % 4 ==0:
        #     train_data_set.clear_cache()
            
        train_data_set.clear_cache()
        yaw = sorted(np.random.choice(range(360), 120))
        print(yaw)
        train_data_set.yaw = yaw
