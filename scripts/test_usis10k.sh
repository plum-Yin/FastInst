
cd /root/autodl-tmp/FastInst

python train_net.py \
  --config-file configs/underwater/instance-segmentation/fastinst_R50_ppm-fpn_x1_576_USIS10K.yaml \
  --eval-only \
  --num-gpus 1 \
  DATASETS.USIS10K_ROOT /root/autodl-tmp/datasets/USIS10K \
  DATASETS.TEST '("usis10k_test",)' \
  MODEL.WEIGHTS output/fastinst_R50_USIS10K/model_best.pth \
  OUTPUT_DIR output/fastinst_R50_USIS10K
