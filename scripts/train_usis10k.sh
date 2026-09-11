
cd /root/autodl-tmp/FastInst

python train_net.py \
  --config-file configs/underwater/instance-segmentation/fastinst_R50_ppm-fpn_x1_576_USIS10K.yaml \
  --num-gpus 1 \
  --resume \
  SOLVER.AMP.ENABLED False \
  DATASETS.USIS10K_ROOT /root/autodl-tmp/datasets/USIS10K \
  OUTPUT_DIR output/fastinst_R50_USIS10K
