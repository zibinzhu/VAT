import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from engineer.utils.geometry import index, orthogonal, back_orthogonal, perspective, back_perspective
from torch.nn import init
class _BasePIFuNet(nn.Module):
    def __init__(self,
                 projection_mode:str='orthogonal',
                 shape_error_term:str='mse',
                 normal_error_term:str='l1',
                 img_error_term:str='l1'
                 ):
        '''
        Parameters
            backbone: Which backbone you use in your PIFu model {Res32|Hourglass ....}
            head: Which function you want to learn, default: iso-surface--->surface_classifier
            depth: The network aims at predict depth of 3-D points
            projection_model : Either orthogonal or perspective.
            param error_term:  nn Loss between the predicted [B, Res, N] and the label [B, Res, N]
        '''

        super(_BasePIFuNet, self).__init__()
        self.__name = 'basePIFu'

        self.shape_error_term = shape_error_term
        self.normal_error_term = normal_error_term
        self.img_error_term = img_error_term
        
        if shape_error_term == 'mse':
            self.shape_error_term = nn.MSELoss()
        elif shape_error_term == 'bce':
            self.shape_error_term = CustomBCELoss(gamma=0.5)
        else:
            raise NotImplementedError

        if normal_error_term == 'l1':
            self.normal_error_term = nn.L1Loss(reduction='sum')
        elif normal_error_term == 'mse':
            self.normal_error_term = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError
        
        if img_error_term == 'l1':
            self.img_error_term = nn.L1Loss(reduction='sum')
        elif img_error_term == 'mse':
            self.img_error_term = nn.MSELoss(reduction='sum')
        else:
            raise NotImplementedError

        self.index = index
        self.projection = orthogonal if projection_mode == 'orthogonal' else perspective
        self.back_projection = back_orthogonal if projection_mode == 'orthogonal' else back_perspective

        self.preds = None
        self.normal_preds = None 
        self.labels = None

    def forward(self, points, images, calibs, transforms=None)->torch.Tensor:
        '''
        Parameters:
            points: [B, 3, N] world space coordinates of points
            images: [B, C, H, W] input images
            calibs: [B, 3, 4] calibration matrices for each image
            transforms: Optional [B, 2, 3] image space coordinate transforms
        Return: 
            [B, Res, N] predictions for each point
        '''

        self.filter(images)
        self.query(points, calibs, transforms)
        return self.get_preds()


    def extract_features(self, images):
        '''
        Filter the input images
        store all intermediate features.

        Parameters:
            images: [B, C, H, W] input images
        '''
        raise NotImplementedError

    def query(self, points, calibs, transforms=None, labels=None):
        '''
        Given 3D points, query the network predictions for each point.
        Image features should be pre-computed before this call.
        store all intermediate features.
        query() function may behave differently during training/testing.

        Parameters:
            points: [B, 3, N] world space coordinates of points
            calibs: [B, 3, 4] calibration matrices for each image
            transforms: Optional [B, 2, 3] image space coordinate transforms
            labels: Optional [B, Res, N] gt labeling
        Return: 
            [B, Res, N] predictions for each point
        '''
        None

    def get_preds(self):
        '''
        Get the predictions from the last query
        :return: [B, Res, N] network prediction for the last query
        '''
        return self.preds

    def get_normal_preds(self):
        '''
        Get the predictions from the last query
        :return: [B, Res, N] network prediction for the last query
        '''
        return self.normal_preds

    def get_error(self):
        '''
        Get the network loss from the last query

        return: 
            loss term
        '''
        return self.error_term(self.preds, self.labels)



    @property
    def name(self):
        __repr = "{}(Parameters: ".format(self.__name)
        for key in self.input_para.keys():
            __repr+="{}:{}, ".format(key,self.input_para[key])
        __repr=__repr[:-2]
        return __repr+')'
    
    @name.setter
    def name(self,v):
        self.__name = v


    @staticmethod
    def init_weights(net, init_type='normal', init_gain=0.02):
        """Initialize network weights.
        Parameters:
            net (network)   -- network to be initialized
            init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

        We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
        work better for some applications. Feel free to try yourself.
        """

        def init_func(m):  # define the initialization function
            classname = m.__class__.__name__
            if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
                if init_type == 'normal':
                    init.normal_(m.weight.data, 0.0, init_gain)
                elif init_type == 'xavier':
                    init.xavier_normal_(m.weight.data, gain=init_gain)
                elif init_type == 'kaiming':
                    init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    init.orthogonal_(m.weight.data, gain=init_gain)
                else:
                    raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
                if hasattr(m, 'bias') and m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif classname.find(
                    'BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
                init.normal_(m.weight.data, 1.0, init_gain)
                init.constant_(m.bias.data, 0.0)

        print('initialize network with %s' % init_type)
        net.apply(init_func)  # apply the initialization function <init_func>

    @staticmethod
    def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
        '''
        Initialize a network:
        Parameters:
            net (network)      -- the network to be initialized
            init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
            gain (float)       -- scaling factor for normal, xavier and orthogonal.
        Return:
            None
        '''
        __class__.init_weights(net, init_type, init_gain=init_gain)


class CustomBCELoss(nn.Module):
    def __init__(self, brock=False, gamma=None, reduction='mean'):
        super(CustomBCELoss, self).__init__()
        self.brock = brock
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, gt, gamma=None, w=None):
        x_hat = torch.clamp(pred, 1e-5, 1.0-1e-5) # prevent log(0) from happening
        gamma = gamma[:,None,None] if self.gamma is None else self.gamma
        if self.brock:
            x = 3.0*gt - 1.0 # rescaled to [-1,2]

            loss = -(gamma*x*torch.log(x_hat) + (1.0-gamma)*(1.0-x)*torch.log(1.0-x_hat))
        else:
            loss = -(gamma*gt*torch.log(x_hat) + (1.0-gamma)*(1.0-gt)*torch.log(1.0-x_hat))

        if w is not None:
            if len(w.size()) == 1:
                w = w[:,None,None] 
            loss = loss * w

        if self.reduction=='sum':
            return loss.sum()
        return loss.mean()
    
class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super(VGGPerceptualLoss, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = nn.ModuleList(blocks)
        self.transform = F.interpolate
        self.mean = nn.Parameter(torch.tensor([0.485, 0.456, 0.406]).view(1,3,1,1))
        self.std = nn.Parameter(torch.tensor([0.229, 0.224, 0.225]).view(1,3,1,1))
        self.resize = resize

    def forward(self, input, target):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
        return loss