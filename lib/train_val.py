import os
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import logging
import argparse
from datetime import datetime
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.model_helper import build_model
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.datasets.rope3d import Rope3D
from lib.datasets.kitti import KITTI


def my_worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def create_logger(log_file):
    log_format = '%(asctime)s  %(levelname)5s  %(message)s'
    # 基础配置：写入文件
    logging.basicConfig(level=logging.INFO, format=log_format, filename=log_file)
    # 控制台输出
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter(log_format))
    logging.getLogger(__name__).addHandler(console)
    return logging.getLogger(__name__)


def get_env_ranks(fallback_local_rank: int = 0):
    """
    统一从环境变量读取分布式参数，兼容 torchrun / slurm / 旧式脚本。
    """
    local_rank = int(os.environ.get("LOCAL_RANK",
                        os.environ.get("SLURM_LOCALID",
                        fallback_local_rank)))
    rank = int(os.environ.get("RANK",
                   os.environ.get("SLURM_PROCID",
                   0)))
    world_size = int(os.environ.get("WORLD_SIZE",
                         os.environ.get("SLURM_NTASKS",
                         torch.cuda.device_count() if torch.cuda.is_available() else 1)))
    return local_rank, rank, world_size


def ddp_init(backend: str = "nccl"):
    """
    用 env:// 初始化分布式。确保 torchrun 自动注入的变量可用。
    """
    if dist.is_available() and not dist.is_initialized():
        dist.init_process_group(backend=backend, init_method="env://")


def is_rank0():
    return int(os.environ.get("RANK", 0)) == 0


def main_worker(args):
    # 解析/获取 ranks
    # 允许命令行给 --local_rank/--local-rank，但优先用环境变量（torchrun）
    cli_local_rank = getattr(args, "local_rank", 0)
    local_rank, rank, world_size = get_env_ranks(cli_local_rank)
    args.local_rank = local_rank
    args.world_size = world_size
    args.rank = rank

    # load cfg
    assert os.path.exists(args.config), f"Config not found: {args.config}"
    with open(args.config, "r") as f:
        cfg = yaml.load(f, Loader=yaml.Loader)

    # 分布式初始化（env://）
    ddp_init(backend="nccl")

    # 设备绑定
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    # 仅 rank0 拷贝 lib 快照
    import shutil
    if not args.evaluate and rank == 0:
        dst_lib = os.path.join(cfg['trainer']['log_dir'], 'lib/')
        os.makedirs(cfg['trainer']['log_dir'], exist_ok=True)
        if os.path.exists(dst_lib):
            shutil.rmtree(dst_lib)
        shutil.copytree('./lib', dst_lib)
    # 等待拷贝完成
    if dist.is_initialized():
        dist.barrier()

    # 随机种子
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        # 可选：提高速度（非确定性）
        # cudnn.benchmark = True

    # logger（各 rank 都建，同一文件；rank0 会有控制台输出 handler）
    os.makedirs(cfg['trainer']['log_dir'], exist_ok=True)
    logger = create_logger(os.path.join(cfg['trainer']['log_dir'], 'train.log'))

    # ===== Dataset & DataLoader =====
    # 根据配置选择数据集类
    dataset_type = cfg['dataset'].get('type', 'rope3d').lower()
    if dataset_type == 'kitti':
        train_set = KITTI(root_dir=cfg['dataset']['root_dir'], split='train', cfg=cfg['dataset'])
    else:
        train_set = Rope3D(root_dir=cfg['dataset']['root_dir'], split='train', cfg=cfg['dataset'])
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,  # 让 sampler 管理 shuffle
        drop_last=False
    )

    # 你原来的 batch 计算：batch_size * 4 / nprocs（保留逻辑）
    per_gpu_bs = int(cfg['dataset']['batch_size'] * 4 / max(world_size, 1))
    train_loader = DataLoader(
        dataset=train_set,
        batch_size=per_gpu_bs,
        num_workers=2,
        pin_memory=True,
        drop_last=False,
        sampler=train_sampler,
        worker_init_fn=my_worker_init_fn
    )

    if dataset_type == 'kitti':
        val_set = KITTI(root_dir=cfg['dataset']['root_dir'], split='val', cfg=cfg['dataset'])
    else:
        val_set = Rope3D(root_dir=cfg['dataset']['root_dir'], split='val', cfg=cfg['dataset'])
    val_loader = DataLoader(
        dataset=val_set,
        batch_size=cfg['dataset']['batch_size'] * 4,
        num_workers=2,
        shuffle=False,
        pin_memory=True,
        drop_last=False
    )

    # ===== Model =====
    model = build_model(cfg['model'], train_loader.dataset.cls_mean_size)

    if args.evaluate:
        # 评估阶段可只在 rank0 跑一次（避免重复评估），也可按需改成各卡分摊
        if rank == 0:
            tester = Tester(cfg, model, val_loader, logger)
            tester.test()
        return

    if torch.cuda.is_available():
        model.cuda(local_rank)

    # DDP 包裹
    model = torch.nn.parallel.DistributedDataParallel(
        model,
        device_ids=[local_rank] if torch.cuda.is_available() else None,
        output_device=local_rank if torch.cuda.is_available() else None,
        find_unused_parameters=True
    )

    # Optimizer & LRScheduler
    optimizer = build_optimizer(cfg['optimizer'], model)
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    # ===== Trainer =====
    trainer = Trainer(
        cfg=cfg,
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        test_loader=val_loader,
        lr_scheduler=lr_scheduler,
        warmup_lr_scheduler=warmup_lr_scheduler,
        logger=logger,
        train_sampler=train_sampler,
        local_rank=local_rank,
        args=args
    )
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='implementation of GUPNet / MonoDETR style training (DDP)')
    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--config', type=str, default='config/config.yaml')
    parser.add_argument('--seed', default=None, type=int, help='seed for initializing training')
    # 兼容两种写法：命令行给了也能接；但实际以环境变量为准（torchrun 会注入）
    parser.add_argument('--local_rank', default=0, type=int, help='(compat) local rank for distributed training')
    parser.add_argument('--local-rank', dest='local_rank', type=int, help='(compat) same as --local_rank')
    # 旧脚本的 ip/端口已不需要；保留参数避免你现有启动脚本报错
    parser.add_argument('--ip', default=1222, type=int, help='(compat) unused when using torchrun')

    # 容忍未知参数，避免启动器额外注入时报错
    args, _ = parser.parse_known_args()

    # world_size/ ranks 在 main_worker 里通过环境变量确定
    main_worker(args)
