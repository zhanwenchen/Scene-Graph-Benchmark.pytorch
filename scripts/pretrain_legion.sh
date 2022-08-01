#!/usr/bin/env bash
  export CUDA_VISIBLE_DEVICES=0,1,2,3
  export NUM_GPUS=4
  export MODEL_NAME="pretrain_resnet_1"
  echo "Started pretraining model ${MODEL_NAME}"
  MODEL_DIRNAME=./checkpoints/pretrained_faster_rcnn/${MODEL_NAME}/
  mkdir ${MODEL_DIRNAME} &&
  cp -r ./tools/ ${MODEL_DIRNAME} &&
  cp -r ./scripts/ ${MODEL_DIRNAME} &&
  cp -r ./maskrcnn_benchmark/ ${MODEL_DIRNAME} &&
  HOST_NODE_ADDR=12345 PYTHONUNBUFFERED=x torchrun --master_port=10001 --nproc_per_node=$NUM_GPUS  tools/detector_pretrain_net.py \
  --config-file "configs/e2e_relation_detector_X_101_32_8_FPN_1x.yaml" \
  MODEL.WEIGHT /home/zhanwen/sgb_og/checkpoints/pretrained_faster_rcnn/X-101-32x8d.pkl \
  MODEL.RELATION_ON False \
  SOLVER.IMS_PER_BATCH 8 \
  TEST.IMS_PER_BATCH ${NUM_GPUS} \
  SOLVER.PRE_VAL False \
  DTYPE "float32" \
  SOLVER.MAX_ITER 50000 \
  SOLVER.STEPS "(30000, 45000)" \
  SOLVER.VAL_PERIOD 2000 \
  SOLVER.CHECKPOINT_PERIOD 2000 \
  GLOVE_DIR ./datasets/vg/ \
  OUTPUT_DIR ${MODEL_DIRNAME} 2>&1 | tee ${MODEL_DIRNAME}/log_train.log
  echo "Finished pretraining model ${MODEL_NAME}"
