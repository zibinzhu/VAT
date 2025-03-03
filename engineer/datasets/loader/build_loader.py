'''
@author: lingteng qiu
@version:1.0
'''
from einops import rearrange
import torch
import torchvision.transforms as transforms


def train_fine_pifu_loader_collate_fn(batches):
    '''collection function of dataloder for training
    '''
    img_list = []
    calib_list = []
    samples_list = []
    labels_list = []
    name_list = []
    front_normal = []
    back_normal = []

    crop_img = []
    crop_front_normal = []
    crop_back_normal = []
    crop_query_points_list = [] 

    for batch in batches:
        name = batch['name']
        image_tensor = batch['img']
        calib_tensor = batch['calib']
        sample_tensor = batch['samples']
        label_tensor = batch['labels']
        front_normal_tensor = batch['front_normal']
        back_normal_tensor = batch['back_normal']
        img_list.append(image_tensor)
        calib_list.append(calib_tensor)
        samples_list.append(rearrange(sample_tensor,'d n -> 1 d n'))
        labels_list.append( rearrange(label_tensor,'d n -> 1 d n'))
        name_list.append(name)
        front_normal.append(front_normal_tensor)
        back_normal.append(back_normal_tensor)

        #crop infos
        crop_query_points = batch['crop_pts']
        crop_query_points_list.append(rearrange(crop_query_points,'d n -> 1 d n'))
        crop_image_tensor = batch['crop_img']
        crop_front_normal_tensor = batch['crop_front_normal']
        crop_back_normal_tensor = batch['crop_back_normal']
        crop_img.append(crop_image_tensor)
        crop_front_normal.append(crop_front_normal_tensor)
        crop_back_normal.append(crop_back_normal_tensor)

    imgs = torch.cat(img_list,dim=0)
    calibs = torch.cat(calib_list,dim=0)
    samples = torch.cat(samples_list,dim=0)
    labels = torch.cat(labels_list,dim=0)
    front_normal = torch.cat(front_normal,dim = 0)
    back_normal = torch.cat(back_normal,dim = 0)

    #crop_infos
    crop_query_points = torch.cat(crop_query_points_list,dim=0)
    crop_img = torch.cat(crop_img,dim=0)
    crop_front_normal = torch.cat(crop_front_normal,dim = 0)
    crop_back_normal = torch.cat(crop_back_normal,dim = 0)

    return dict(name=name_list,img=imgs,calib=calibs,samples=samples,labels=labels, front_normal = front_normal,back_normal =back_normal, \
        crop_img = crop_img,crop_front_normal = crop_front_normal,crop_back_normal = crop_back_normal,crop_query_points=crop_query_points)

def test_fine_pifu_loader_collate_fn(batches):
    '''collection function of dataloder for training
    '''
    img_list = []
    calib_list = []
    samples_list = []
    labels_list = []
    name_list = []
    front_normal = []
    back_normal = []

    crop_img = []
    crop_front_normal = []
    crop_back_normal = []
    crop_query_points_list = [] 

    for batch in batches:
        name = batch['name']
        image_tensor = batch['img']
        calib_tensor = batch['calib']
        sample_tensor = batch['samples']
        label_tensor = batch['labels']
        front_normal_tensor = batch['front_normal']
        back_normal_tensor = batch['back_normal']
        img_list.append(image_tensor)
        calib_list.append(calib_tensor)
        samples_list.append(rearrange(sample_tensor,'d n -> 1 d n'))
        labels_list.append( rearrange(label_tensor,'d n -> 1 d n'))
        name_list.append(name)
        front_normal.append(front_normal_tensor)
        back_normal.append(back_normal_tensor)

        #crop infos
        crop_query_points = batch['crop_pts']
        crop_query_points_list.append(rearrange(crop_query_points,'d n -> 1 d n'))
        crop_image_tensor = batch['crop_img']
        crop_front_normal_tensor = batch['crop_front_normal']
        crop_back_normal_tensor = batch['crop_back_normal']
        crop_img.append(crop_image_tensor)
        crop_front_normal.append(crop_front_normal_tensor)
        crop_back_normal.append(crop_back_normal_tensor)

    imgs = torch.cat(img_list,dim=0)
    calibs = torch.cat(calib_list,dim=0)
    samples = torch.cat(samples_list,dim=0)
    labels = torch.cat(labels_list,dim=0)
    front_normal = torch.cat(front_normal,dim = 0)
    back_normal = torch.cat(back_normal,dim = 0)

    #crop_infos
    crop_query_points = torch.cat(crop_query_points_list,dim=0)
    crop_img = torch.cat(crop_img,dim=0)
    crop_front_normal = torch.cat(crop_front_normal,dim = 0)
    crop_back_normal = torch.cat(crop_back_normal,dim = 0)

    return dict(name=name_list,img=imgs,calib=calibs,samples=samples,labels=labels, front_normal = front_normal,back_normal =back_normal, \
        crop_img = crop_img,crop_front_normal = crop_front_normal,crop_back_normal = crop_back_normal,crop_query_points=crop_query_points)

def train_loader_collate_fn(batches):
    '''collection function of dataloder for training
    '''
    img_list = []
    calib_list = []
    samples_list = []
    labels_list = []
    name_list = []
    front_normal = []
    back_normal = []

    for batch in batches:
        name = batch['name']
        image_tensor = batch['img']
        calib_tensor = batch['calib']
        sample_tensor = batch['samples']
        label_tensor = batch['labels']
        front_normal_tensor = batch['front_normal']
        back_normal_tensor = batch['back_normal']
        img_list.append(image_tensor)
        calib_list.append(calib_tensor)
        samples_list.append(rearrange(sample_tensor,'d n -> 1 d n'))
        labels_list.append( rearrange(label_tensor,'d n -> 1 d n'))
        name_list.append(name)
        front_normal.append(front_normal_tensor)
        back_normal.append(back_normal_tensor)

    imgs = torch.cat(img_list,dim=0)
    calibs = torch.cat(calib_list,dim=0)
    samples = torch.cat(samples_list,dim=0)
    labels = torch.cat(labels_list,dim=0)
    front_normal = torch.cat(front_normal,dim = 0)
    back_normal = torch.cat(back_normal,dim = 0)

    return dict(name=name_list,img=imgs,calib=calibs,samples=samples,labels=labels, front_normal = front_normal,back_normal =back_normal)

def test_loader_collate_fn(batches):
    '''collection function of dataloder for testing
    '''
    img_list = []
    calib_list = []
    samples_list = []
    labels_list = []
    name_list = []
    front_normal = []
    back_normal = []

    for batch in batches:
        name = batch['name']
        image_tensor = batch['img']
        calib_tensor = batch['calib']
        sample_tensor = batch['samples']
        label_tensor = batch['labels']
        front_normal_tensor = batch['front_normal']
        back_normal_tensor = batch['back_normal']
        img_list.append(image_tensor)
        calib_list.append(calib_tensor)
        samples_list.append(rearrange(sample_tensor,'d n -> 1 d n'))
        labels_list.append(rearrange(label_tensor,'d n -> 1 d n'))
        name_list.append(name)
        front_normal.append(front_normal_tensor)
        back_normal.append(back_normal_tensor)

    imgs = torch.cat(img_list,dim=0)
    calibs = torch.cat(calib_list,dim=0)
    samples = torch.cat(samples_list,dim=0)
    labels = torch.cat(labels_list,dim=0)
    front_normal = torch.cat(front_normal,dim = 0)
    back_normal = torch.cat(back_normal,dim = 0)

    return dict(name=name_list,img=imgs,calib=calibs,samples=samples,labels=labels, front_normal = front_normal,back_normal =back_normal)

def train_mvspifu_loader_collate_fn(batches):
    '''collection function of dataloder for training
    '''
    # ray_o_list = []
    ray_map_list = []
    img_list = []
    mask_list = []
    calib_list = []
    samples_list = []
    labels_list = []
    name_list = []

    for batch in batches:
        name = batch['name']
        # ray_o_tensor = batch['ray_o']
        ray_map_tensor = batch['ray_map']
        image_tensor = batch['img']
        #normal_tensor = batch['normal']
        #mask_tensor = batch['mask']
        calib_tensor = batch['calib']
        sample_tensor = batch['samples']
        label_tensor = batch['labels']

        # ray_o_list.append(ray_o_tensor)
        ray_map_list.append(ray_map_tensor)
        img_list.append(image_tensor)
        #normal_list.append(normal_tensor)
        #mask_list.append(mask_tensor)
        calib_list.append(calib_tensor)
        samples_list.append(rearrange(sample_tensor,'d n -> 1 d n'))
        labels_list.append(rearrange(label_tensor,'d n -> 1 d n'))
        name_list.append(name)

    # ray_os = torch.cat(ray_o_list,dim=0)
    ray_maps = torch.cat(ray_map_list,dim=0)
    imgs = torch.cat(img_list,dim=0)
    #normals = torch.cat(normal_list,dim=0)
    #masks = torch.cat(mask_list,dim=0)
    calibs = torch.cat(calib_list,dim=0)
    samples = torch.cat(samples_list,dim=0)
    labels = torch.cat(labels_list,dim=0)

    return dict(name=name_list,ray_map=ray_maps,img=imgs,calib=calibs,samples=samples,labels=labels)

def test_mvspifu_loader_collate_fn(batches):
    '''collection function of dataloder for training
    '''
    ray_map_list = []
    # ray_o_list = []
    img_list = []
    #mask_list = []
    calib_list = []
    samples_list = []
    labels_list = []
    name_list = []
    yid_list = []

    for batch in batches:
        name = batch['name']
        yid = batch['yid']
        ray_map_tensor = batch['ray_map']
        image_tensor = batch['img']
        #normal_tensor = batch['normal']
        mask_tensor = batch['mask']
        calib_tensor = batch['calib']
        sample_tensor = batch['samples']
        label_tensor = batch['labels']

        name_list.append(name)
        yid_list.append(yid)
        ray_map_list.append(ray_map_tensor)
        # ray_o_list.append(ray_o_tensor)
        img_list.append(image_tensor)
       # normal_list.append(normal_tensor)
        #mask_list.append(mask_tensor)
        calib_list.append(calib_tensor)
        samples_list.append(rearrange(sample_tensor,'d n -> 1 d n'))
        labels_list.append(rearrange(label_tensor,'d n -> 1 d n'))
        

    # ray_os = torch.cat(ray_o_list,dim=0)
    ray_maps = torch.cat(ray_map_list,dim=0)
    imgs = torch.cat(img_list,dim=0)
    #normals = torch.cat(normal_list,dim=0)
    #masks = torch.cat(mask_list,dim=0)
    calibs = torch.cat(calib_list,dim=0)
    samples = torch.cat(samples_list,dim=0)
    labels = torch.cat(labels_list,dim=0)

    return dict(name=name_list,yid=yid_list,ray_map=ray_maps,img=imgs,calib=calibs,samples=samples,labels=labels)

def carton_test_loader_collate_fn(batches):
    

    '''collection function of dataloder for testing
    '''
    img_list = []
    name_list = []

    for batch in batches:
        img = batch['img']
        img_list.append(img)
        name_list.append(batch['name'])

    imgs = torch.cat(img_list,dim=0)

    return dict(name=name_list,img=imgs)