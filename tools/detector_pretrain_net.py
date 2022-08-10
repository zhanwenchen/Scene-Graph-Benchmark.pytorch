# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
r"""
Basic training script for PyTorch
"""

# Set up custom environment before nearly anything else is imported
# NOTE: this should be the first import (no not reorder)
from comet_ml import API, Experiment, ExistingExperiment
from maskrcnn_benchmark.utils.env import setup_environment  # noqa F401 isort:skip

from math import sqrt
import argparse
from time import time as time_time
from os import environ as os_environ
from os.path import join as os_path_join
from datetime import timedelta as datetime_timedelta
from torch import (
    as_tensor as torch_as_tensor,
    cat as torch_cat,
    device as torch_device,
    no_grad as torch_no_grad,
)
from torch.cuda import max_memory_allocated, set_device
from torch.nn.parallel import DistributedDataParallel
from torch.nn.utils import clip_grad_norm_
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.distributed import init_process_group
# from apex.amp import scale_loss as amp_scale_loss, initialize as amp_initialize
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data import make_data_loader
from maskrcnn_benchmark.solver import make_lr_scheduler
from maskrcnn_benchmark.solver import make_optimizer
from maskrcnn_benchmark.engine.trainer import reduce_loss_dict
from maskrcnn_benchmark.engine.inference import inference
from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.utils.collect_env import collect_env_info
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, all_gather
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.metric_logger import MetricLogger
from maskrcnn_benchmark.utils.comet import get_experiment
from maskrcnn_benchmark.utils.utils_train import setup_seed
from util_misc import load_gbnet_vgg_weights, load_gbnet_fcs_weights, load_gbnet_rpn_weights


APEX_FUSED_OPTIMIZERS = {'FusedSGD', 'FusedAdam'}
OPTIMIZERS_WITH_SCHEDULERS = {'SGD', 'FusedSGD'}


def train(cfg, local_rank, distributed, logger, experiment):
    batch_size = cfg.SOLVER.IMS_PER_BATCH
    model = build_detection_model(cfg)
    device = torch_device(cfg.MODEL.DEVICE)
    model.to(device, non_blocking=True)

    using_scheduler = cfg.SOLVER.TYPE in OPTIMIZERS_WITH_SCHEDULERS
    optimizer, lrs_by_name = make_optimizer(cfg, model, logger, rl_factor=int(os_environ.get("NUM_GPUS", 1)) * sqrt(batch_size), return_lrs_by_name=True)
    hyperparameters = {'batch_size': batch_size, **lrs_by_name}
    if not isinstance(experiment, ExistingExperiment):
        experiment.log_parameters(hyperparameters)
    print('hyperparameters =', hyperparameters)
    using_reduce_lr_on_plateau = cfg.MODEL.BACKBONE.CONV_BODY == 'VGG-16' and using_scheduler
    if using_reduce_lr_on_plateau:
        scheduler = ReduceLROnPlateau(optimizer, 'max', patience=3, factor=0.1,
            verbose=True, threshold=0.001, threshold_mode='abs', cooldown=1)
    else:
        scheduler = make_lr_scheduler(cfg, optimizer) if using_scheduler else None

    output_dir = cfg.OUTPUT_DIR
    arguments = {}
    save_to_disk = get_rank() == 0
    logger.info('instatntiating checkpointer')
    checkpointer = DetectronCheckpointer(
        cfg, model, optimizer, scheduler, output_dir, save_to_disk
    )
    if not cfg.MODEL.BACKBONE.CONV_BODY == 'VGG-16':
        logger.info('finished instatntiating checkpointer')
        extra_checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, update_schedule=cfg.SOLVER.UPDATE_SCHEDULE_DURING_LOAD)
        logger.info('finished loading extra checkpoint data')
        arguments.update(extra_checkpoint_data)

    if cfg.MODEL.BACKBONE.CONV_BODY == 'VGG-16':
        vgg16_pretrain_strategy = cfg.MODEL.VGG.PRETRAIN_STRATEGY
        fpath = cfg.MODEL.VGG.GBNET_PRETRAINED_DETECTOR_FPATH
        if vgg16_pretrain_strategy == 'none':
            pass
        elif vgg16_pretrain_strategy == 'backbone':
            state_dict = load_gbnet_vgg_weights(model, fpath)
            del state_dict
        elif vgg16_pretrain_strategy == 'fcs':
            state_dict = load_gbnet_vgg_weights(model, fpath)
            state_dict = load_gbnet_fcs_weights(model, fpath, state_dict=state_dict)
            del state_dict
        elif vgg16_pretrain_strategy == 'rpn':
            state_dict = load_gbnet_vgg_weights(model, fpath)
            state_dict = load_gbnet_fcs_weights(model, fpath, state_dict=state_dict)
            state_dict = load_gbnet_rpn_weights(model, fpath, state_dict=state_dict)
            del state_dict
        else:
            raise NotImplementedError(f'vgg16_pretrain_strategy={vgg16_pretrain_strategy} is not a valid strategy')
    model.to(device, non_blocking=True)

    # Initialize mixed-precision training
    use_mixed_precision = cfg.DTYPE == "float16"
    # amp_opt_level = 'O1' if use_mixed_precision else 'O0'
    # model, optimizer = amp_initialize(model, optimizer, opt_level=amp_opt_level)

    if distributed:
        logger.info('starting distributed')
        model = DistributedDataParallel(
            model, device_ids=[local_rank], output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=False,
        )
        logger.info('ending distributed')

    arguments["iteration"] = 0
    logger.info('making data loaders')
    train_data_loader = make_data_loader(
        cfg,
        mode='train',
        is_distributed=distributed,
        start_iter=arguments["iteration"],
    )
    val_data_loaders = make_data_loader(
        cfg,
        mode='val',
        is_distributed=distributed,
    )
    logger.info('finished making data loaders')
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD

    if cfg.SOLVER.PRE_VAL:
        with experiment.validate():
            logger.info("Validate before training")
            loss_val = run_val(cfg, model, val_data_loaders, distributed)
            experiment.log_metric('loss', loss_val, epoch=0)
            experiment.log_epoch_end(0)

    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(train_data_loader)
    start_iter = arguments["iteration"]
    start_training_time = time_time()
    end = time_time()
    clip = cfg.SOLVER.GRAD_NORM_CLIP
    for iteration, (images, targets, _) in enumerate(train_data_loader, start_iter):
        with experiment.train():
            model.train()

            if any(len(target) < 1 for target in targets):
                logger.error(f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}" )
            data_time = time_time() - end
            iteration = iteration + 1
            arguments["iteration"] = iteration

            images = images.to(device, non_blocking=True)
            targets = [target.to(device) for target in targets]

            loss_dict = model(images, targets)

            losses = sum(loss for loss in loss_dict.values())

            # reduce losses over all GPUs for logging purposes
            loss_dict_reduced = reduce_loss_dict(loss_dict)
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            meters.update(loss=losses_reduced, **loss_dict_reduced)
            experiment.log_metrics(loss_dict_reduced, epoch=iteration)

            if cfg.SOLVER.TYPE in APEX_FUSED_OPTIMIZERS:
                optimizer.zero_grad() # For Apex FusedSGD, FusedAdam, etc
            else:
                optimizer.zero_grad(set_to_none=True) # For Apex FusedSGD, FusedAdam, etc
            # Note: If mixed precision is not used, this ends up doing nothing
            # Otherwise apply loss scaling for mixed-precision recipe
            # with amp_scale_loss(losses, optimizer) as scaled_losses:
            #     scaled_losses.backward()

            losses.backward()

            clip_grad_norm_(model.parameters(), clip)

            optimizer.step()

            batch_time = time_time() - end
            end = time_time()
            meters.update(time=batch_time, data=data_time)

            eta_seconds = meters.time.global_avg * (max_iter - iteration)
            eta_string = str(datetime_timedelta(seconds=int(eta_seconds)))

            if iteration % 200 == 0 or iteration == max_iter:
                lr_i = optimizer.param_groups[0]["lr"]
                logger.info(
                    meters.delimiter.join(
                        [
                            "eta: {eta}",
                            "iter: {iter}",
                            "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=lr_i,
                        memory=max_memory_allocated() / 1048576.0,
                    )
                )
                experiment.log_metric('lr', lr_i)

        if cfg.SOLVER.TO_VAL and iteration % cfg.SOLVER.VAL_PERIOD == 0:
            with experiment.validate():
                logger.info("Start validating")
                mAP = run_val(cfg, model, val_data_loaders, distributed)
                experiment.log_metric('mAP', mAP, epoch=iteration)
                if using_scheduler:
                    scheduler.step(mAP)
        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

        experiment.log_epoch_end(iteration)

    total_training_time = time_time() - start_training_time
    total_time_str = str(datetime_timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )

    return model


@torch_no_grad()
def run_val(cfg, model, val_data_loaders, distributed):
    if distributed:
        model = model.module
    #torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations", )
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes", )

    dataset_names = cfg.DATASETS.VAL
    val_result = []
    for dataset_name, val_data_loader in zip(dataset_names, val_data_loaders):
        dataset_result = inference(
            cfg,
            model,
            val_data_loader,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=None,
        )
        synchronize()
        val_result.append(dataset_result)
    gathered_result = all_gather(torch_as_tensor(dataset_result).cpu())
    gathered_result = [t.view(-1) for t in gathered_result]
    gathered_result = torch_cat(gathered_result, dim=-1).view(-1)
    valid_result = gathered_result[gathered_result>=0]
    del gathered_result
    # from evaluate: float(np.mean(result_dict[mode + '_recall'][100]))
    val_result = float(valid_result.mean())
    #torch.cuda.empty_cache()
    return val_result


@torch_no_grad()
def run_test(cfg, model, distributed):
    if distributed:
        model = model.module
    #torch.cuda.empty_cache()  # TODO check if it helps
    iou_types = ("bbox",)
    if cfg.MODEL.MASK_ON:
        iou_types = iou_types + ("segm",)
    if cfg.MODEL.KEYPOINT_ON:
        iou_types = iou_types + ("keypoints",)
    if cfg.MODEL.RELATION_ON:
        iou_types = iou_types + ("relations", )
    if cfg.MODEL.ATTRIBUTE_ON:
        iou_types = iou_types + ("attributes", )
    output_folders = [None] * len(cfg.DATASETS.TEST)
    dataset_names = cfg.DATASETS.TEST
    if cfg.OUTPUT_DIR:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os_path_join(cfg.OUTPUT_DIR, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(
        cfg,
        mode='test',
        is_distributed=distributed
        )
    val_result = []
    for output_folder, dataset_name, data_loader_val in zip(output_folders, dataset_names, data_loaders_val):
        dataset_result = inference(
            cfg,
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False if cfg.MODEL.RETINANET_ON else cfg.MODEL.RPN_ONLY,
            device=cfg.MODEL.DEVICE,
            expected_results=cfg.TEST.EXPECTED_RESULTS,
            expected_results_sigma_tol=cfg.TEST.EXPECTED_RESULTS_SIGMA_TOL,
            output_folder=output_folder,
        )
        synchronize()
        val_result.append(dataset_result)
    gathered_result = all_gather(torch_as_tensor(dataset_result).cpu())
    gathered_result = [t.view(-1) for t in gathered_result]
    gathered_result = torch_cat(gathered_result, dim=-1).view(-1)
    valid_result = gathered_result[gathered_result>=0]
    del gathered_result
    # from evaluate: float(np.mean(result_dict[mode + '_recall'][100]))
    val_result = float(valid_result.mean())
    #torch.cuda.empty_cache()
    return val_result


def main():
    setup_seed(1234)
    parser = argparse.ArgumentParser(description="PyTorch Object Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    num_gpus = int(os_environ.get("WORLD_SIZE", 1))
    args.distributed = num_gpus > 1

    if args.distributed:
        set_device(args.local_rank)
        init_process_group(
            backend="nccl", init_method="env://"
        )
        synchronize()

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("maskrcnn_benchmark", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Collecting env info (might take some time)")
    logger.info("\n" + collect_env_info())

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    output_config_path = os_path_join(cfg.OUTPUT_DIR, 'config.yml')
    logger.info("Saving config into: {}".format(output_config_path))
    # save overloaded model config in the output directory
    save_config(cfg, output_config_path)

    # Create an experiment with your api key
    model_name = os_environ['MODEL_NAME']
    experiment = get_experiment(model_name)
    experiment.set_name(model_name)
    model = train(cfg, args.local_rank, args.distributed, logger, experiment)

    if not args.skip_test:
        with experiment.test():
            mAP = run_test(cfg, model, args.distributed)
            experiment.log_metric('mAP', mAP, epoch=cfg.SOLVER.MAX_ITER)

if __name__ == "__main__":
    main()
