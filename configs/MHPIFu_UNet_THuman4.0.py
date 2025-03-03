'''
training fine PIFu with gt normal map
'''
import numpy as np
import torch
"---------------------------- debug options -----------------------------"
debug = False
"---------------------------- normal options -----------------------------"
normal = False
fine_pifu = False
"---------------------------- views options -----------------------------"
"---------------------------- views options -----------------------------"
num_views = 3
img_size = 1024
"----------------------------- Model options -----------------------------"
in_channels=3
if normal:
    in_channels=6

model = dict(
    PIFu = dict(
        type = 'HRMVPIFuNet',
        feat_init = dict(type='InitialNet', in_channels=in_channels, hidden_channels=64, fine_pifu=fine_pifu),

        mvpifu = dict( 
            PIFu = dict(
                type='MVPIFuNet',
                unet = dict(type='HRUNetNeck', hidden_channels=64),
                low_attention = dict(type='AttentionDecoder', n_layers=2, q_model=283, k_model=283, v_model=256, d=32, n_head=8, d_inner=256),
                cross_attention = dict(type='AttentionDecoder', n_layers=2, q_model=265, k_model=475, v_model=475, d=32, n_head=8, d_inner=256),
                pe_position = dict(type='Embedder', include_input=True, input_dims=1, max_freq_log2=3, num_freqs=4, log_sampling=True, periodic_fns=[torch.sin, torch.cos]),
                pe_ray = dict(type='Embedder', include_input=True, input_dims=3, max_freq_log2=3, num_freqs=4, log_sampling=True, periodic_fns=[torch.sin, torch.cos]),
                coarse_head =dict(type='PIFuhd_Surface_Head', filter_channels=[283, 1024, 512, 256, 128, 1], merge_layer=2, res_layers=[1, 2, 3, 4], norm=None, last_op='sigmoid'),
                fine_head =dict(type='PIFuhd_Surface_Head', filter_channels=[265, 512, 256, 128, 1], merge_layer=-1, res_layers=[1, 2], norm=None, last_op='sigmoid'),
            ),
            pretrain_weights=None
        ),

        projection_mode ='perspective',
        shape_error_term = 'bce',
        color_error_term = 'mse',
        num_views = num_views,
        img_size = img_size
    )
)
"----------------------------- Datasets options -----------------------------"
dataset_type = 'THuman2.0'
data = dict(
    train = dict(
    type = "TH4Dataset",
    input_dir = '/data/zhuzibin/Workspace/Python/Dataset/THuman/THuman2.0_1024',
    b_min = np.array([-128, -128, -128]), 
    b_max = np.array([128, 128, 128]),
    #b_min = np.array([-128, -28, -128]),
    #b_max = np.array([128, 228, 128]),    
    is_train = True,
    random_multiview = False,
    img_size = img_size,
    num_views = num_views,
    num_sample_points = 8192, 
    num_sample_color = 0,
    sample_sigma = [5., 4.],
    check_occ = 'trimesh',
    debug = debug,
    span = 1,
    normal = normal,
    fine_pifu = fine_pifu,
    test = False
    ),
    test = dict(
    type = "TH4Dataset",
    input_dir = '/data/zhuzibin/Workspace/Python/Dataset/THuman/THuman4.0/THuman4.0',
    # b_min = np.array([-1.5, -1.5, -3.5]),
    # b_max = np.array([3.5, 3.5, 1.5]),
    b_min = np.array([-0.5, -1.5, -2.5]),
    b_max = np.array([2.5, 1.5, 0.5]),
    is_train = False,
    random_multiview = False,
    img_size = img_size,
    num_views = num_views,
    num_sample_points = 0, 
    num_sample_color = 0,
    sample_sigma = [5., 4.],
    check_occ = 'trimesh',
    debug = debug,
    span = 1,
    normal = normal,
    fine_pifu = fine_pifu,
    test = True
    )
)
train_collect_fn = 'train_mvspifu_loader_collate_fn'
test_collect_fn = 'test_mvspifu_loader_collate_fn'
"----------------------------- checkpoints options -----------------------------"
checkpoints = "./checkpoints"
logger = True
num_gpu = 2
lr_policy="stoneLR"
lr_warm_up = 5e-5
warm_epoch = 3
LR=5e-4
num_epoch= 60
batch_size = 8
test_batch_size = 1
scheduler=dict(
    gamma = 0.1,
    stone = [50] 
)
"----------------------------- optimizer options -----------------------------"
optim_para=dict(
    optimizer = dict(type='RMSprop', lr=LR, momentum=0, weight_decay=0.0000),
    # optimizer = dict(type='adam', lr=LR),
)
"----------------------------- training strategies -----------------------------"
save_fre_epoch = 1
"----------------------------- evaluation setting -------------------------------"
val_epoch = 1
start_val_epoch = 0
"----------------------------- inference setting -------------------------------"
resolution = 512 #for inference
"-------------------------------- config name --------------------------------"
name='MHPIFu_UNet_THuman4.0_test'
"-------------------------------- render --------------------------------"
render_cfg = dict(
    type='Noraml_Render',
    width = 512,
    height = 512,
    render_lib ='face3d',
    flip =True
)