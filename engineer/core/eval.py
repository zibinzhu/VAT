import  numpy as np
from collections import defaultdict
import json
from tqdm import tqdm
import cv2
import os
import torch
import copy
from engineer.utils.metrics import *
from utils.structure import AverageMeter
from utils.distributed import reduce_tensor
import time
import logging 
import torch.distributed as dist
from engineer.utils.mesh_utils import gen_mesh, render
import torch.nn.functional as F
logger = logging.getLogger('logger.trainer')
import numpy as np 

def inference(model, cfg, args, test_loader, epoch, gallery_id, gallery_time=0):
    model.eval()
    is_render = False
    start_time = time.time()
    i = 0
    for batch in test_loader:
        
        yid = batch['yid'][0]
        name = batch['name'][0]
        print('name: '+name, 'yid: '+str(yid))
        img = batch['img']
        # normal = batch['normal']
        # ray_o = batch['ray_o']
        ray_map = batch['ray_map']
        #mask = batch['mask']
        calib = batch['calib']
        samples = batch['samples']

        B_MIN = cfg.data.test.b_min
        B_MAX = cfg.data.test.b_max

        # B_MIN = np.array([-0.5, -0.5, -0.5])
        # B_MAX = np.array([0.5, 0.5, 0.5])

        save_gallery_path = gallery_id
        os.makedirs(save_gallery_path,exist_ok=True)

        data = {
            'name': name,
            'yid': yid,
            'img': img,
            # 'normal': normal,
            'ray_map': ray_map,
            # 'ray_o': ray_o,
            'calib': calib,
            #'mask': mask,
            'samples': samples,
            'b_min': B_MIN,
            'b_max': B_MAX,
        }
        if not is_render:
            with torch.no_grad():
                gen_mesh(cfg,model,data,save_gallery_path)
        else:
            render(cfg,model,data,save_gallery_path)
        i+=1
    end_time  = time.time()
    print(f"运行时间: {end_time - start_time} 秒, 平均用时： {(end_time - start_time)/i}")

def test_epoch(model, cfg, args, test_loader, epoch, gallery_id):
    '''test epoch
    Parameters:
        model:
        cfg:
        args:
        test_loader:
        epoch: current epoch
        gallery_id: gallery save path
    Return:
        test_metrics
    '''
    model.eval()
    
    iou_metrics = AverageMeter()
    prec_metrics = AverageMeter()
    recall_metrics = AverageMeter()
    error_metrics =AverageMeter()
    c_loss = AverageMeter()
    f_loss = AverageMeter()
    n_loss = AverageMeter()
    epoch_start_time = time.time()
    
    with torch.no_grad():
        for idx,data in enumerate(test_loader): 
            #ray_o = data['ray_o'].cuda() 
            ray_map = data['ray_map'].cuda() 
            img = data['img'].cuda()
            #mask = data['mask'].cuda()
            calib = data['calib'].cuda()
            samples = data['samples'].cuda()
            labels = data['labels'].cuda()
            # normal = data['normal'].cuda()
            # if cfg.normal:
            #     img = torch.cat([img, normal], dim=1)

            # res, error, closs, floss, nloss  = model(images=img, normals=normal, ray_map=ray_map, masks=mask, points=samples, calibs=calib, labels=labels)
            res, error = model(images=img, ray_map=ray_map, points=samples, calibs=calib, labels=labels)

            """ error = error.sum()
            closs = closs.sum()
            floss = floss.sum()
            nloss = nloss.sum() """
            error = error.sum()
                
            IOU, prec, recall = compute_acc(res,labels)
            if args.dist:
                error = reduce_tensor(error)
                IOU = reduce_tensor(IOU)
                prec =  reduce_tensor(prec)
                recall =  reduce_tensor(recall) 
            error_metrics.update(error.item()/cfg.num_gpu,cfg.test_batch_size)
            """ c_loss.update(closs.item()/cfg.num_gpu, cfg.test_batch_size)
            f_loss.update(floss.item()/cfg.num_gpu, cfg.test_batch_size)
            n_loss.update(nloss.item()/cfg.num_gpu, cfg.test_batch_size) """
            iou_metrics.update(IOU.item(),cfg.test_batch_size)
            prec_metrics.update(prec.item(),cfg.test_batch_size)
            recall_metrics.update(recall.item(),cfg.test_batch_size)

            iter_net_time = time.time()
            eta = int(((iter_net_time - epoch_start_time) / (idx + 1)) * len(test_loader) - (
                    iter_net_time - epoch_start_time))

            word_handler = 'Test: {0} | Epoch: {1} | {2}/{3} | Err: {4:.06f} | cErr: {5:.06f} | fErr: {6:.06f} | nErr: {7:.06f} | IOU: {8:.06f}  | prec: {9:.05f} | recall: {10:.05f}'.format( 
                cfg.name,epoch,idx,len(test_loader),error_metrics.avg,c_loss.avg,f_loss.avg,n_loss.avg,iou_metrics.avg, 
                prec_metrics.avg,recall_metrics.avg, 
                int(eta//3600),int((eta%3600)//60),eta%60)
            if (not args.dist) or dist.get_rank() == 0:
                logger.info(word_handler)
        logger.info("Test Final result | Epoch: {0:d} | Err: {1:.06f} | IOU: {2:.06f}  | prec: {3:.05f} | recall: {4:.05f} |".format(
            epoch, error_metrics.avg, iou_metrics.avg, prec_metrics.avg, recall_metrics.avg
            ))
    return dict(error=error_metrics.avg,iou=iou_metrics.avg,recall = recall_metrics.avg,pre =prec_metrics.avg)

            

            


    


