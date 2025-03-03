import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .BasePIFu import _BasePIFuNet
from engineer.models.registry import PIFU
from engineer.utils.train_utils import reshape_sample_tensor
from engineer.models.builder import build_backbone, build_head


@PIFU.register_module
class SRPIFuNet(_BasePIFuNet):

    def __init__(self,unet,low_attention,cross_attention,pe_position,pe_ray,coarse_head,fine_head,
                 projection_mode:str='orthogonal',shape_error_term:str='mse',img_error_term:str='l1',num_views:int=1,img_size:int=512):
        super(SRPIFuNet, self).__init__(projection_mode, shape_error_term, img_error_term)

        self.name = 'SRPIFuNet'

        self.feature_extractor = build_backbone(unet)
        self.low_attention = build_backbone(low_attention)
        self.cross_attention = build_backbone(cross_attention)
        self.pe_position = build_backbone(pe_position)
        self.pe_ray = build_backbone(pe_ray)
        self.coarse_head = build_head(coarse_head)
        self.fine_head = build_head(fine_head)

        self.num_views = num_views
        self.is_render = False
        self.img_size = img_size
        self.border_radius = 128.0

        self.input_para =dict(
            feature_extractor = self.feature_extractor.name,
            low_attention = self.low_attention.name,
            cross_attention = self.cross_attention.name,
            pe_position = self.pe_position.name,
            pe_ray = self.pe_ray.name,
            coarse_head = self.coarse_head.name,
            fine_head = self.fine_head.name,
            projection_mode = projection_mode,
            shape_error_term = shape_error_term,
            num_views = num_views
        )

        self.init_net(self)

    def get_feat(self, images, ray_map):     
        self.low_img_feats, self.sub_img_feats, self.hight_img_feats = self.feature_extractor(images)
        self.ray_map = ray_map
    
    def get_uv(self, points, calibs, transforms=None):
        points = reshape_sample_tensor(points, self.num_views)
        xyz_list = self.projection(points, calibs, transforms)
        self.xy_list = xyz_list[:, :2, :]
        self.xy_list = (self.xy_list - self.img_size / 2) / (self.img_size / 2)#([6, 2, 8192])
        self.in_img = (self.xy_list[:, 0] >= -1.0) & (self.xy_list[:, 0] <= 1.0) & (self.xy_list[:, 1] >= -1.0) & (self.xy_list[:, 1] <= 1.0)
        B, C, N = points.shape
        B =  B // self.num_views
        self.in_img = self.in_img.reshape(B, self.num_views, 1, N)
        self.xyz = torch.mean(points.reshape(B, self.num_views, C, N), dim=1)
        # self.xyz[:, 1, :] = self.xyz[:, 1, :] + 0.2
        # self.xyz[:, 1, :] = self.xyz[:, 1, :] - 100
        self.xyz = self.xyz / self.border_radius

    def self_attention(self, feature_fusion, features):
        B, V, C, N = features.shape
        features = features.permute(0, 3, 1, 2).contiguous().reshape(-1, V, C)
        features, = feature_fusion(features)
        features = features.reshape(B, N, V, C).permute(0, 2, 3, 1)
        return features

    def attention(self, feature_fusion, query_features, key_features, value_features):
        B, V, C_q, N = query_features.shape
        B, V, C_k, N = key_features.shape
        B, V, C_v, N = value_features.shape
        query_features = query_features.permute(0, 3, 1, 2).contiguous().reshape(-1, V, C_q)
        key_features = key_features.permute(0, 3, 1, 2).contiguous().reshape(-1, V, C_k)
        value_features = value_features.permute(0, 3, 1, 2).contiguous().reshape(-1, V, C_v)

        query_features = feature_fusion(query_features, key_features, value_features)
        query_features = query_features.reshape(B, N, V, C_q).permute(0, 2, 3, 1)
        return query_features

    def coarse_query(self):
        #8192*256*3                   128*128*256*3        8192*2（N*2）
        low_point_feats = self.index(self.low_img_feats, self.xy_list)
        point_ray_feats = self.index(self.ray_map, self.xy_list)
        point_ray_feats = self.pe_ray(point_ray_feats.permute(0, 2, 1)).permute(0, 2, 1)
        #8192*283*3==>8192*283*1
        point_ray_feats = torch.cat([low_point_feats, point_ray_feats], dim=-2)
        B, C, N = point_ray_feats.shape
        B = B // self.num_views
        point_ray_feats = point_ray_feats.reshape(B, self.num_views, C, N)

        _, C, N = low_point_feats.shape
        low_point_feats = low_point_feats.reshape(B, self.num_views, C, N)
        
        self.low_point_feats = self.attention(self.low_attention, point_ray_feats, point_ray_feats, low_point_feats)

        self.coarse_preds, self.low_point_target_feats = self.coarse_head(self.low_point_feats[:, 1])

        self.coarse_preds = self.coarse_preds.unsqueeze(1).repeat(1, self.num_views, 1, 1)
        self.coarse_preds = self.in_img.float() * self.coarse_preds
        self.coarse_preds = torch.mean(self.coarse_preds, dim=1)
        self.preds = self.coarse_preds
    
    def fine_query(self):
        hight_point_feats = self.index(self.hight_img_feats, self.xy_list)
        sub_point_feats = self.index(self.sub_img_feats, self.xy_list)
        hight_point_feats = torch.cat([sub_point_feats, hight_point_feats], dim=-2)

        B, C, N = hight_point_feats.shape
        B = B // self.num_views
        hight_point_feats = hight_point_feats.reshape(B, self.num_views, C, N)

        hight_point_feats = torch.cat([self.low_point_feats, hight_point_feats], dim=-2)

        hight_point_feats = self.attention(self.cross_attention, self.low_point_target_feats, hight_point_feats, hight_point_feats)[:, 1]
        self.fine_preds, _ = self.fine_head(hight_point_feats)

        self.fine_preds = self.fine_preds.unsqueeze(1).repeat(1, self.num_views, 1, 1)
        self.fine_preds = self.in_img.float() * self.fine_preds
        self.fine_preds = torch.mean(self.fine_preds, dim=1)
        self.preds = self.fine_preds

    def fine_query_mean(self):
        hight_point_feats = self.index(self.hight_img_feats, self.xy_list)
        sub_point_feats = self.index(self.sub_img_feats, self.xy_list)
        hight_point_feats = torch.cat([sub_point_feats, hight_point_feats], dim=-2)

        B, C, N = hight_point_feats.shape
        B = B // self.num_views
        hight_point_feats = hight_point_feats.reshape(B, self.num_views, C, N)
        hight_point_feats = torch.cat([self.low_point_feats, hight_point_feats], dim=-2)
        hight_point_feats  = torch.mean(hight_point_feats , dim=1)

        self.fine_preds, _ = self.fine_head(hight_point_feats)

        self.fine_preds = self.fine_preds.unsqueeze(1).repeat(1, self.num_views, 1, 1)
        self.fine_preds = self.in_img.float() * self.fine_preds
        self.fine_preds = torch.mean(self.fine_preds, dim=1)
        self.preds = self.fine_preds

    def get_error(self, labels):
        cross_error = self.shape_error_term(self.coarse_preds, labels)
        fine_error = self.shape_error_term(self.fine_preds, labels)
        return cross_error+fine_error

    def forward(self, images, ray_map, points, calibs, transforms=None, labels=None):

        self.get_feat(images, ray_map)

        self.get_uv(points=points, calibs=calibs, transforms=transforms)

        self.coarse_query()

        self.fine_query()

        res = self.get_preds()

        error = self.get_error(labels=labels)

        return res, error