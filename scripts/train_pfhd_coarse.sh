# coarse-pifuhd
# CUDA_VISIBLE_DEVICES=6 python ./tools/train_pifu.py --current 0   --config ./configs/PIFuhd_Render_People_HG_coarse.py
CUDA_VISIBLE_DEVICES=0,2,4,6 python -m torch.distributed.launch --nproc_per_node=4 ./tools/train_pifu.py --dist --current 0 --config ./configs/PIFuhd_Render_People_HG_coarse.py