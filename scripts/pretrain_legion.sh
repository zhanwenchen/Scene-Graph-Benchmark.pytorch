#!/usr/bin/env bash
  export CUDA_VISIBLE_DEVICES=4,5,6,7
  export NUM_GPUS=4
  export MODEL_NAME="pretrain_vgg_og_comet_4"
  echo "Started pretraining model ${MODEL_NAME}"
  MODEL_DIRNAME=./checkpoints/pretrained_faster_rcnn/${MODEL_NAME}/
  mkdir ${MODEL_DIRNAME} &&
  cp -r ./tools/ ${MODEL_DIRNAME} &&
  cp -r ./scripts/ ${MODEL_DIRNAME} &&
  cp -r ./maskrcnn_benchmark/ ${MODEL_DIRNAME} &&
  python -m torch.distributed.launch --master_port=10001 --nproc_per_node=$NUM_GPUS  tools/detector_pretrain_net.py \
  --config-file "configs/pretrain_detector_VGG16_1x.yaml" \
  MODEL.VGG.PRETRAIN_STRATEGY backbone \
  MODEL.RELATION_ON False \
  SOLVER.TYPE "Adam" \
  SOLVER.IMS_PER_BATCH 64 \
  TEST.IMS_PER_BATCH ${NUM_GPUS} \
  SOLVER.PRE_VAL True \
  DTYPE "float32" \
  SOLVER.MAX_ITER 50000 \
  SOLVER.VAL_PERIOD 2000 \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  GLOVE_DIR ./datasets/vg/ \
  OUTPUT_DIR ${MODEL_DIRNAME} 2>&1 | tee ${MODEL_DIRNAME}/log_train.log
  echo "Finished pretraining model ${MODEL_NAME}"
