
cd /root/autodl-tmp/FastInst

python train_net.py \
  --config-file configs/underwater/instance-segmentation/fastinst_R50_ppm-fpn_x1_576_UIIS.yaml \
  --eval-only \
  --num-gpus 1 \
  DATASETS.UIIS_ROOT /root/autodl-tmp/datasets/UIIS \
  MODEL.WEIGHTS output/fastinst_R50_UIIS/model_best.pth \
  OUTPUT_DIR output/fastinst_R50_UIIS
