from lib.config import cfg, args
from lib.networks import make_network
from lib.train import make_trainer, make_optimizer, make_lr_scheduler, make_recorder, set_lr_scheduler
from lib.datasets import make_data_loader
from lib.utils.net_utils import load_model, save_model, load_network, load_pretrain
from lib.evaluators import make_evaluator
import torch.multiprocessing
import torch
import torch.distributed as dist
import os
import torch
from torch.utils.data import Subset, SubsetRandomSampler
import numpy as np

def get_subset_loader(val_loader, subset_ratio=0.1):
    # Lấy dataset từ val_loader
    dataset = val_loader.dataset
    
    # Tính toán số lượng mẫu cần thiết
    total_size = len(dataset)
    subset_size = int(total_size * subset_ratio)
    
    # Tạo danh sách các chỉ số cho subset
    indices = list(range(total_size))
    subset_indices = torch.randperm(total_size).tolist()[:subset_size]

    # Tạo sampler cho subset
    subset_sampler = SubsetRandomSampler(subset_indices)
    
    # Tạo DataLoader cho subset
    subset_loader = DataLoader(dataset,
                              sampler=subset_sampler,
                              batch_size=val_loader.batch_size,
                              num_workers=val_loader.num_workers,
                              collate_fn=val_loader.collate_fn,
                              worker_init_fn=val_loader.worker_init_fn)
    
    return subset_loader

if cfg.fix_random:
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train(cfg, network):
    train_loader = make_data_loader(cfg,
                                    is_train=True,
                                    is_distributed=cfg.distributed,
                                    max_iter=cfg.ep_iter)

    if cfg.skip_eval:
        val_loader = None
    else:
        val_loader = make_data_loader(cfg, is_train=False)
    trainer = make_trainer(cfg, network, train_loader)
    optimizer = make_optimizer(cfg, network)
    scheduler = make_lr_scheduler(cfg, optimizer)
    recorder = make_recorder(cfg)
    evaluator = make_evaluator(cfg)

    begin_epoch = load_model(network,
                             optimizer,
                             scheduler,
                             recorder,
                             cfg.trained_model_dir,
                             resume=cfg.resume)
    if begin_epoch == 0 and cfg.pretrain != '':
        load_pretrain(network, cfg.pretrain)

    set_lr_scheduler(cfg, scheduler)
    psnr_best, ssim_best, lpips_best = 0, 0, 10
    for epoch in range(begin_epoch, cfg.train.epoch):
        recorder.epoch = epoch
        if cfg.distributed:
            train_loader.batch_sampler.sampler.set_epoch(epoch)

        train_loader.dataset.epoch = epoch

        trainer.train(epoch, train_loader, optimizer, recorder)
        scheduler.step()

        if (epoch + 1) % cfg.save_ep == 0 and cfg.local_rank == 0:
            save_model(network, optimizer, scheduler, recorder,
                       cfg.trained_model_dir, epoch)
            
        if (epoch + 1) % cfg.save_latest_ep == 0 and cfg.local_rank == 0:
            save_model(network,
                       optimizer,
                       scheduler,
                       recorder,
                       cfg.trained_model_dir,
                       epoch,
                       last=True)

        if not cfg.skip_eval and (epoch + 1) % cfg.eval_ep == 0 and cfg.local_rank == 0:
            result = trainer.val(epoch, val_loader, evaluator, recorder)
            psnr = result['psnr']
            ssim = result['ssim']
            lpips = result['lpips']
            if psnr > psnr_best:
                psnr_best = psnr
                save_model(network,
                       optimizer,
                       scheduler,
                       recorder,
                       cfg.trained_model_dir,
                       epoch,
                       custom='psnr_best')
            if ssim > ssim_best:
                ssim_best = ssim
                save_model(network,
                       optimizer,
                       scheduler,
                       recorder,
                       cfg.trained_model_dir,
                       epoch,
                       custom='ssim_best')
            if lpips < lpips_best:
                lpips_best = lpips
                save_model(network,
                       optimizer,
                       scheduler,
                       recorder,
                       cfg.trained_model_dir,
                       epoch,
                       custom='lpips_best')
            print(f'psnr_best: {psnr_best:.2f}, ssim_best: {ssim_best:.3f}, lpips_best: {lpips_best:.3f}')

            

    return network



def test(cfg, network):
    trainer = make_trainer(cfg, network)
    val_loader = make_data_loader(cfg, is_train=False)
    # print("val_loader", len(val_loader))
    # val_loader_sub = get_subset_loader(val_loader)
    # print("val_loader_sub",len(val_loader_sub))
    evaluator = make_evaluator(cfg)
    epoch = load_network(network,
                         cfg.trained_model_dir,
                         resume=cfg.resume,
                         epoch=cfg.test.epoch)
    trainer.val(epoch, val_loader, evaluator)

def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()

def main():
    if cfg.distributed:
        cfg.local_rank = int(os.environ['RANK']) % torch.cuda.device_count()
        torch.cuda.set_device(cfg.local_rank)
        torch.distributed.init_process_group(backend="nccl",
                                             init_method="env://")
        synchronize()

    network = make_network(cfg)
    if args.test:
        test(cfg, network)
    else:
        train(cfg, network)
    if cfg.local_rank == 0:
        print('Success!')
        print('='*80)
    os.system('kill -9 {}'.format(os.getpid()))


if __name__ == "__main__":
    main()
