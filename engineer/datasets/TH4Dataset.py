'''
@author: lingteng qiu
@data  : 2021-1-21
@emai: lingtengqiu@link.cuhk.edu.cn
RenderPeople Dataset:https://renderpeople.com/
'''
import cv2
import sys
sys.path.append("./")
from torch.utils.data import Dataset
import json
import os
import numpy as np
import random
import torch
import scipy.sparse as sp
from .pipelines import Compose
from .registry import DATASETS
from engineer.utils.libmesh import check_mesh_contains
import warnings
from PIL import Image,ImageOps
from PIL.ImageFilter import GaussianBlur
from multiprocessing import Process, Manager, Lock
import torchvision.transforms as transforms
import trimesh
import numpy as np
import math
from tqdm import tqdm
import logging
import torch.nn.functional as F
logger = logging.getLogger('logger.trainer')


@DATASETS.register_module
class TH4Dataset(Dataset):
    def __init__(self,input_dir, b_min, b_max, is_train=True, projection_mode='orthogonal', random_multiview=False, img_size=512, num_views=1, total_samples=1000000, num_sample_points=5000, \
        num_sample_color=0, sample_sigma=5., check_occ='trimesh', debug=False, span=1, normal=False, fine_pifu = False, crop_windows=512, test=False, is_blur=True):
        '''
        Render People Dataset
        Parameters:
            input_dir: file direction e.g. Garmant/render_people_gen, in this file you have some subfile direction e.g. rp_kai_posed_019_BLD
            caceh: memeory cache which is employed to save sample points from mesh. Of course, we use it to speed up data loaded. 
            pipeline: the method which process datasets, like crop, ColorJitter and so on.
            is_train: phase the datasets' state
            projection_mode: orthogonal or perspective
            num_sample_points: the number of sample clounds from mesh 
            num_sample_color: the number of sample colors from mesh, default 0, means train shape model
            sample_sigma: the distance we disturb points sampled from surface. unit: cm e.g you wanna get 5cm, you need input 5
            check_occ: which method, you use it to check whether sample points are inside or outside of mesh. option: trimesh |
            debug: debug the dataset like project the points into img_space scape
            span: span step from 0 deg to 359 deg, e.g. if span == 2, deg: 0 2 4 6 ...,
            normal: whether, you want to use normal map, default False, if you want to train pifuhd, you need set it to 'True'
            sample_aim: set sample distance from mesh, according to PIFu it use 5 cm while PIFuhd choose 5 cm for coarse PIFu and 3 cm
                to fine PIFu
            fine_pifu: whether train fine pifu,
            crop_windows: crop window size using for pifuhd, default, 512
            test: whether it is test-datasets 
        Return:
            None
        '''
        super(TH4Dataset,self).__init__()
        self.is_train = is_train
        self.projection_mode = projection_mode
        self.__name="THuamn2.0"
        self.img_size = img_size
        self.num_views = num_views
        self.__B_MIN = b_min
        self.__B_MAX = b_max
        self.border_radius = math.sqrt(3*(b_max[0]**2))
        self.total_samples = total_samples
        self.num_sample_points = num_sample_points
        self.num_sample_color = num_sample_color
        self.sigma = sample_sigma if type(sample_sigma) == list else [sample_sigma]
        #view from render
        self.root = input_dir
        self.RENDER = os.path.join(self.root, 'RENDER')
        self.RENDER_NORMAL = os.path.join(self.root, 'NORMAL_PRED2')
        self.MASK = os.path.join(self.root, 'MASK')
        self.PARAM = os.path.join(self.root, 'PARAM')
        self.OBJ = os.path.join(self.root, 'OBJ')
        self.DIVIDED = os.path.join(self.root, 'DIVIDED')

        self.cache_data = Manager().dict()
        self.cache_data_lock = Lock()

        if test:
            self.yaw = [0, 2, 4, 6]
            # self.yaw = [0, 2, 4, 6]
        else:
            self.yaw = sorted(np.random.choice(range(360), 72))
        print(self.yaw)
        
        self.yaw_list = list(range(0,24,span))
        self.__pitch_list = [0]

        angle_field = 60
        self.range_angle = range(-angle_field,angle_field)
        self.range_angle_weight = []
        for i in self.range_angle:
            if i<0:
                angle_weight = (i + angle_field)**2 / (angle_field**2*2)
            else:
                angle_weight = (-i + angle_field)**2 / (angle_field**2*2)
            self.range_angle_weight.append(angle_weight)
        self.range_angle_weight /= np.sum(self.range_angle_weight)
        self.random_offset = list(range(-60,60))

        self.normal = normal
        self.subjects = self.get_subjects()
        self.random_multiview = random_multiview
        self.check_occ = check_occ
        self.debug = debug
        self.span = span
        self.fine_pifu = fine_pifu
        self.crop_windows_size = crop_windows
        self.test = test 

        self.is_blur = is_blur
        
        
        if self.test:
            self.is_train = False

        self.input_para=dict(
            root = input_dir,
            is_train = is_train,
            projection_mode = projection_mode,
            img_size = img_size,
            num_views = num_views,
            num_sample_points = num_sample_points,
            num_sample_color = num_sample_color,
            random_multiview = random_multiview,
            sample_sigma=sample_sigma,
            check_occ=check_occ,
            debug = debug,
            span = span,
            normal = normal,
            fine_pifu = fine_pifu,
            crop_windows_size = crop_windows,
            test = test 
        )

        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        self.aug_trans = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.3, hue=0.05)
        ])
    
    def clear_cache(self):
        self.cache_data.clear()
    
    def get_index(self,index):
        sid = index % len(self.subjects)
        tmp = index // len(self.subjects)
        yid = tmp % len(self.yaw)
        pid = tmp // len(self.yaw)
        return sid,yid,pid 
    
    def check_sigma_path(self,sigma_path_list):
        '''check whether all sigma_path exites 
        '''
        for sigma_path in sigma_path_list:
            if not os.path.exists(sigma_path):
                return False
        return True

    def visibility_sample(self, data, res):
        sample_points = data['sample_points']
        inside = data['inside']

        num_sample_points = self.num_sample_points
        if self.test:
            num_sample_points = self.num_sample_points*5
        
        inside_points = sample_points[inside]
        np.random.shuffle(inside_points)
        outside_points = sample_points[np.logical_not(inside)]
        np.random.shuffle(outside_points)

        nin = inside_points.shape[0]
        inside_points = inside_points[
                        :num_sample_points // 2] if nin >num_sample_points // 2 else inside_points
        outside_points = outside_points[
                            :num_sample_points // 2] if nin > num_sample_points // 2 else outside_points[:(num_sample_points - nin)]

        samples = np.concatenate([inside_points, outside_points], 0).T
        labels = np.concatenate([np.ones((1, inside_points.shape[0])), np.zeros((1, outside_points.shape[0]))],1)

        samples = torch.Tensor(samples).float()
        labels = torch.Tensor(labels).float()
        
        return {
            'samples': samples,
            'labels': labels
        }

    def select_sampling_method(self, subject):
        if self.cache_data.__contains__(subject):
            return self.cache_data[subject]
        mesh = trimesh.load(os.path.join(self.OBJ, subject, '%s.obj' % subject))
        
        surface_points_1, _ = trimesh.sample.sample_surface(mesh, 64 * self.num_sample_points)
        sample_points_1 = surface_points_1 + np.random.normal(scale=self.sigma[0], size=surface_points_1.shape)

        """ surface_points_2, _ = trimesh.sample.sample_surface(mesh, 16 * self.num_sample_points)
        sample_points_2 = surface_points_2 + np.random.normal(scale=self.sigma[1], size=surface_points_2.shape) """

        length = self.B_MAX - self.B_MIN
        random_points = np.random.rand(4 * self.num_sample_points, 3) * length + self.B_MIN
        sample_points = np.concatenate([sample_points_1, random_points], 0)
        inside = check_mesh_contains(mesh, sample_points)

        del mesh

        self.cache_data_lock.acquire()
        self.cache_data[subject] = {
            'sample_points': sample_points,
            'inside': inside,
        }
        self.cache_data_lock.release()
        print(subject, self.cache_data.__len__())
        return self.cache_data[subject]

    def get_subjects(self):
        if self.is_train:
            train_subjects = np.loadtxt(os.path.join(self.DIVIDED, 'train.txt'), dtype=str)
            print('train_obj_size: %i' % len(train_subjects))
            return sorted(list(train_subjects))
        else:
            var_subjects = np.loadtxt(os.path.join(self.DIVIDED, 'val.txt'), dtype=str)
            print('test_obj_size: %i' % len(var_subjects))
            return sorted(list(var_subjects))
    
    def get_rays(self, original_img_size, img_size, K, c2w):
        factor = img_size / original_img_size 
        K = K * factor
        i, j = np.meshgrid(np.arange(img_size, dtype=np.float32), np.arange(img_size, dtype=np.float32), indexing='xy')
        dirs = np.stack([(i-K[0][2])/K[0][0], (j-K[1][2])/K[1][1], np.ones_like(i)], -1)
        # Rotate ray directions from camera frame to the world frame
        rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = np.broadcast_to(c2w[:3,-1], np.shape(rays_d))
        rays_d = rays_d / np.linalg.norm(rays_d, axis=-1, keepdims=True)
        return rays_d

    def gen_full_rays(self, extrinsic, intrinsic, resolution):
        rot = extrinsic[:3, :3].transpose(0, 1)
        trans = -torch.mm(rot, extrinsic[:3, 3:])
        c2w = torch.cat((rot, trans), dim=1)

        fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
        W = resolution[0]
        H = resolution[1]
        i, j = torch.meshgrid(torch.linspace(0.5, W-0.5, W, device=c2w.device), torch.linspace(0.5, H-0.5, H, device=c2w.device))  # pytorch's meshgrid has indexing='ij'
        i = i.t()
        j = j.t()
        dirs = torch.stack([(i-cx)/fx, (j-cy)/fy, torch.ones_like(i)], -1)
        # Rotate ray directions from camera frame to the world frame
        rays_d = torch.sum(dirs[..., np.newaxis, :] * c2w[:3,:3], -1)  # dot product, equals to: [c2w.dot(dir) for dir in dirs]
        # Translate camera frame's origin to the world frame. It is the origin of all rays.
        rays_o = c2w[:3,-1]

        rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

        return rays_d, rays_o

    def get_render(self, subject, num_views, yid=0):
        yid = self.yaw[yid]
        view_ids = [self.yaw_list[(yid + len(self.yaw_list) // num_views * offset) % len(self.yaw_list)]
                    for offset in range(num_views)]
        if self.is_train:
            random_offset = random.sample(range(-20,19), 2)
            random_offset = np.insert(random_offset, 0, 0)
            view_ids = np.add(view_ids, random_offset)
            view_ids = view_ids % 360
        print(view_ids)
        calib_list = []
        render_list = []
        extrinsic_list = []
        normal_list = []
        mask_list = []
        rays_o_list = []
        rays_map_list = []
    
        for vid in view_ids:
            extrinsic_path = os.path.join(self.PARAM, subject, '{}_extrinsic.npy'.format(vid))
            intrinsic_path = os.path.join(self.PARAM, subject, '{}_intrinsic.npy'.format(vid))
            # 修
            # render_path = os.path.join(self.RENDER, subject, '{}.png'.format(vid))
            # normal_path = os.path.join(self.RENDER_NORMAL, subject, '{}.png'.format(vid))
            # mask_path = os.path.join(self.MASK, subject, '{}.png'.format(vid))
            render_path = os.path.join(self.RENDER, subject, '{}.jpg'.format(vid))
            normal_path = os.path.join(self.RENDER_NORMAL, subject, '{}.png'.format(vid))
            mask_path = os.path.join(self.MASK, subject, '{}.jpg'.format(vid))

            extrinsic = np.load(extrinsic_path)
            intrinsic = np.load(intrinsic_path)
            mask = Image.open(mask_path).convert('L')
            render = Image.open(render_path).convert('RGB')
            normal = render

            if self.normal:
                normal = Image.open(normal_path)
            imgs_list = [render, normal, mask]

            # random flip
            if self.is_train and np.random.rand() > 0.5:
                for i, img in enumerate(imgs_list):
                    imgs_list[i] = transforms.RandomHorizontalFlip(p=1.0)(img)
                intrinsic[0, :] *= -1.0
                intrinsic[0, 2] += self.img_size
            
            # 修
            # intrinsic[1, :] *= -1.0
            # intrinsic[1, 2] += self.img_size

            if self.is_train:
                # Pad images
                pad_size = int(0.1 * self.img_size)
                for i, img in enumerate(imgs_list):
                    imgs_list[i] = ImageOps.expand(img, pad_size, fill=0)

                w, h = imgs_list[0].size
                th, tw = self.img_size, self.img_size

                # random scale
                rand_scale = random.uniform(0.9, 1.1)
                w = int(rand_scale * w)
                h = int(rand_scale * h)
                for i, img in enumerate(imgs_list):
                    imgs_list[i] = img.resize((w, h), Image.BILINEAR)
                intrinsic[0, 0] *= rand_scale
                intrinsic[1, 1] *= rand_scale

                # random translate in the pixel space
                dx = random.randint(-int(round((w - tw) / 10.)),
                                    int(round((w - tw) / 10.)))
                dy = random.randint(-int(round((h - th) / 10.)),
                                    int(round((h - th) / 10.)))

                intrinsic[0, 2] += -dx
                intrinsic[1, 2] += -dy

                x1 = int(round((w - tw) / 2.)) + dx
                y1 = int(round((h - th) / 2.)) + dy

                for i, img in enumerate(imgs_list):
                    imgs_list[i] = img.crop((x1, y1, x1 + tw, y1 + th))

                render, normal, mask = imgs_list
                render = self.aug_trans(render)

                blur = GaussianBlur(np.random.uniform(0, 0.5))
                render = render.filter(blur)
                if self.normal:
                    normal = normal.filter(blur)

            calib = torch.Tensor(np.matmul(intrinsic, extrinsic)).float()

            intrinsic = torch.Tensor(intrinsic).float()
            extrinsic = torch.Tensor(extrinsic).float()

            # 修 self.img_size
            rays_map, rays_o = self.gen_full_rays(extrinsic, intrinsic, [1330, 1150])
            rays_map = rays_map.permute(2, 1, 0)

            mask = torch.FloatTensor(np.array(mask)).permute(1, 0) / 255.0
            mask = mask.reshape(1, 1330, 1150)
            mask[mask >= 0.5] = 1.0

            mask = mask.repeat(3, 1, 1)

            p2d = (90, 90, 0, 0)
            mask = F.pad(mask, p2d, 'constant', 0)
            
            # 修 维度变了
            render = self.to_tensor(render).permute(0, 2, 1)
            normal = self.to_tensor(normal).permute(0, 2, 1)

            render = F.pad(render, p2d, 'constant', 0)
            # normal = F.pad(normal, p2d, 'constant', 0)
            rays_map = F.pad(rays_map, p2d, 'constant', 0)


            render = render * mask
            
            
            # 修 
            """ if self.img_size > 512 and not self.fine_pifu:
                render = F.interpolate(torch.unsqueeze(render, 0), size=(512, 512), mode='bilinear', align_corners=True)[0]
                normal = F.interpolate(torch.unsqueeze(normal, 0), size=(512, 512), mode='bilinear', align_corners=True)[0]
                mask = F.interpolate(torch.unsqueeze(mask, 0), size=(512, 512), mode='bilinear', align_corners=True)[0] """
            render = F.interpolate(torch.unsqueeze(render, 0), size=(768, 768), mode='bilinear', align_corners=True)[0]
            # normal = F.interpolate(torch.unsqueeze(normal, 0), size=(768, 768), mode='bilinear', align_corners=True)[0]
            mask = F.interpolate(torch.unsqueeze(mask, 0), size=(768, 768), mode='bilinear', align_corners=True)[0]

            normal = normal * mask

            render = render.permute(0, 2, 1)
            normal = normal.permute(0, 2, 1)
            mask = mask.permute(0, 2, 1)

            rays_o_list.append(rays_o)
            rays_map_list.append(rays_map)
            render_list.append(render)
            calib_list.append(calib)
            extrinsic_list.append(extrinsic)
            normal_list.append(normal)
            mask_list.append(mask)

        return {
            'ray_o': torch.stack(rays_o_list, dim=0),
            'ray_map': torch.stack(rays_map_list, dim=0),
            'img': torch.stack(render_list, dim=0),
            'calib': torch.stack(calib_list, dim=0),
            'extrinsic': torch.stack(extrinsic_list, dim=0),
            'normal': torch.stack(normal_list, dim=0),
            'mask': torch.stack(mask_list, dim=0),
        }

    @property
    def pitch(self):
        return self.__pitch_list

    @property
    def B_MAX(self):
        return self.__B_MAX
    @property
    def B_MIN(self):
        return self.__B_MIN

    def __repr__(self):
        __repr = "{}(Parameters: ".format(self.__name)
        for key in self.input_para.keys():
            __repr+="{}:{}, ".format(key,self.input_para[key])
        __repr=__repr[:-2]
        return __repr+')'
    
    def __getitem__(self,index):
        sid,yid,pid = self.get_index(index)
        
        subject = self.subjects[sid]
        res = {
        'name': subject,
        'mesh_path': os.path.join(self.OBJ, subject, subject + '.obj'),
        'mesh_dir': os.path.join(self.OBJ, subject),
        'sid': sid,
        'yid': self.yaw[yid],
        'pid': pid,
        'b_min': self.B_MIN,
        'b_max': self.B_MAX,
        }

        render_data = self.get_render(subject, num_views=self.num_views, yid=yid)
        res.update(render_data)

        if self.num_sample_points:
            sample_data = self.select_sampling_method(subject)
            sample_data = self.visibility_sample(sample_data, res)
            res.update(sample_data)
        else:
            sample_data = {
                'samples': torch.zeros((3,2)),
                'labels': torch.zeros((1,2))
            }
            res.update(sample_data)
        
        return res

    def __len__(self):   
        return len(self.subjects) * len(self.yaw) * len(self.pitch)