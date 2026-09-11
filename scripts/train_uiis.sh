
cd /root/autodl-tmp/FastInst

python train_net.py \
  --config-file configs/underwater/instance-segmentation/fastinst_R50_ppm-fpn_x1_576_UIIS.yaml \
  --num-gpus 1 \
  SOLVER.AMP.ENABLED False \
  DATASETS.UIIS_ROOT /root/autodl-tmp/datasets/UIIS \
  OUTPUT_DIR output/fastinst_R50_UIIS 
