import os

import torch
import torch.distributed as dist
from hydra import compose
from hydra.utils import instantiate
from sam2.build_sam import build_sam2
from torch import nn
from training.utils.utils import count_trainable_params
from utils import *
import logging

from training.optimizer import construct_optimizer
from training.trainer import CudaConf, DistributedConf, OptimAMPConf, OptimConf
from training.utils.data_utils import BatchedVideoDatapoint
from training.utils.logger import setup_logging
from training.utils.train_utils import (
    AverageMeter,
    get_amp_type,
    is_dist_avail_and_initialized,
)


def record_loss_meters(loss_meters, losses, device):
    # Record the losses
    for k in losses:
        for key in losses[k]:
            if key not in loss_meters:
                loss_meters[key] = AverageMeter(f"Train_{k}_{key}", device)
            loss_meters[key].update(losses[k][key].item())


def train_on_epoch(
    cfg, model, train_dataloader, criterion, scaler, optim, device, epoch, loss_meters
):
    model.train()
    for data_iter, batch in enumerate(train_dataloader):
        # print(batch)
        batch: BatchedVideoDatapoint = batch
        optim.zero_grad(set_to_none=True)
        # with torch.cuda.amp.autocast(
        #                 enabled=optim_conf.amp.enabled,
        #                 dtype=get_amp_type(optim_conf.amp.amp_dtype),
        #                     ):
        outputs = model(batch.to(device, non_blocking=True))
        targets = batch.masks.to(device, non_blocking=True)
        losses = {key: criterion[key](outputs, targets) for key in criterion}

        exact_epoch = epoch + float(data_iter) / len(train_dataloader)
        where = float(exact_epoch) / 200
        assert where <= 1 + 1e-8, f"where: {where}"
        if where < 1.0:
            optim.step_schedulers(where, step=int(exact_epoch * len(train_dataloader)))
        core_loss = losses["all"]["core_loss"]

        record_loss_meters(loss_meters, losses, device)

        # core_loss.backward()
        scaler.scale(core_loss).backward()
        scaler.step(optim.optimizer)
        scaler.update()
        if data_iter % 10 == 0:
            logging.info(f"Epoch: {epoch}, Iter: {data_iter}, Loss: {core_loss.item()}")


@torch.inference_mode()
def evaluate(model, val_dataloader, device):
    model.eval()
    for data_iter, batch in enumerate(val_dataloader):
        with torch.no_grad():
            batch = batch.to(device, non_blocking=True)
            outputs = model(batch)
            targets = batch.masks
            logging.info(data_iter)
    return None


def init_distributed_mode(local_rank, world_size):
    gpu_num = world_size
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(55662)
    os.environ["RANK"] = str(local_rank)
    os.environ["LOCAL_RANK"] = str(local_rank)
    os.environ["WORLD_SIZE"] = str(gpu_num)
    cuda_conf = CudaConf()
    distributed_conf = DistributedConf()

    distributed_conf.backend = "nccl"
    rank = setup_torch_dist_and_backend(cuda_conf, distributed_conf)
    assert is_dist_avail_and_initialized(), "分布式环境未初始化"
    setup_for_distributed(rank == 0)  # 设置为rank0打印日志
    return rank


def main(local_rank, world_size, args):

    rank = init_distributed_mode(local_rank, world_size)
    setup_logging(__name__, rank=rank, output_dir=".")
    # 加载配置文件
    cfg = compose(config_name=args.config)
    optim_conf = OptimConf(**cfg.trainer.optim)
    device = setup_device("cuda")

    # 初始化数据加载器
    data = instantiate(cfg.trainer.data)
    dataset_num = data["train"].datasets.__len__()

    logging.info(f"gpu_num: {world_size}")
    logging.info(f"数据集数量: {dataset_num}")
    for d in range(dataset_num):
        logging.info(f"数据集 {d} 数量: {data['train'].datasets[d].__len__()}")

    # 构建模型
    model = build_sam2(
        "configs/sam2.1/sam2.1_hiera_b+.yaml", ckpt_path="bplus_finetune.pth"
    )
    #    multitask_num=len(cfg.trainer.data.train.datasets[0].dataset.datasets[0].video_dataset.gt_folder))

    scaler = torch.amp.GradScaler(
        device,
        enabled=optim_conf.amp.enabled if optim_conf else False,
    )
    # gradient_clipper = (
    #         instantiate(optim_conf.gradient_clip) if optim_conf else None
    #     )
    optim_conf.options["lr"][0]["scheduler"]["start_value"] = 1.0e-5
    optim_conf.options["lr"][0]["scheduler"]["end_value"] = 1.0e-6
    optim_conf.options["lr"][1]["scheduler"]["start_value"] = 1.0e-5
    optim_conf.options["lr"][1]["scheduler"]["end_value"] = 1.0e-6
    # 构建优化器
    optim = construct_optimizer(
        model,
        optim_conf.optimizer,
        optim_conf.options,
        optim_conf.param_group_modifiers,
    )
    param_info = count_trainable_params(model)

    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=True,  # 如果模型里有条件分支，可改 True
    )
    loss = {
        key: el  # wrap_base_loss(el)
        for (key, el) in instantiate(cfg.trainer.loss, _convert_="all").items()
    }
    loss = nn.ModuleDict(loss)
    # val_loader = data["val"].get_loader(0)
    for epoch in range(args.epochs):
        train_loader = data["train"].get_loader(epoch)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()
        loss_meters = {}
        train_on_epoch(
            cfg, model, train_loader, loss, scaler, optim, device, epoch, loss_meters
        )
        loss_meters = f"Train[{epoch}/{args.epochs}]: " + ", ".join(
            [str(v) for k, v in loss_meters.items()]
        )
        # logging.info(loss_meters)
        logging.info(loss_meters)
        # evaluate(model, val_loader, device)
    checkpoint = {"model": model.module.state_dict()}
    torch.save(checkpoint, "bplus_finetune_mul_query.pth")
    logging.info("训练完成")


def parse_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/sam2.1_training/sam2.1_hiera_b+_trop2_mul_query.yaml",
        help="config file",
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--devices", type=str, default="0,1,2,3")
    parser.add_argument("--epochs", type=int, default=200)

    return parser.parse_args()


if __name__ == "__main__":
    import warnings

    warnings.filterwarnings("ignore")
    args = parse_args()
    local_rank = args.local_rank

    torch.multiprocessing.set_start_method(
        "spawn"
    )  # CUDA runtime does not support `fork`
    os.environ["CUDA_VISIBLE_DEVICES"] = args.devices
    world_size = len(os.environ["CUDA_VISIBLE_DEVICES"].replace(" ", "").split(","))

    if world_size == 1:
        main(local_rank, world_size, args)
    else:
        mp_runner = torch.multiprocessing.start_processes
        _args = (
            world_size,
            args,
        )
        mp_runner(main, args=_args, nprocs=world_size, start_method="spawn")
