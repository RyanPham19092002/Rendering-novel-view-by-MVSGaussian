import numpy as np
import imageio.v2 as imageio
import os
from glob import glob
from lib.config import cfg
from lib.datasets import mvsgs_utils
import torch
from lib.utils import data_utils
import argparse
import yaml
from lib.networks import make_network
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from tqdm import tqdm

zfar = 100.0
znear = 0.01
trans = [0.0, 0.0, 0.0]
scale = 1.0

def read_cam_file(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    # extrinsics: line [1,5), 4x4 matrix
    print(filename)
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
    extrinsics = extrinsics.reshape((4, 4))
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
    intrinsics = intrinsics.reshape((3, 3))
    # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0])
    depth_max = float(lines[11].split()[1])
    return intrinsics, extrinsics, depth_min, depth_max


def read_config(config):
    with open(config, 'r') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
        cfg.mvsgs.cas_config.volume_planes = result['volume_planes']
        cfg.mvsgs.reweighting = result['reweighting']
        scale_factor = result['scale_factor']
        cfg.mvsgs.eval_center = result['eval_center']
        cfg.mvsgs.cas_config.render_if = result['render_if']
    return scale_factor


def load_data(src_dir, tgt_dir, scale_factor):
        ret = {}
        all_batches = []

    # # Duyệt qua các thư mục con
    # subfolders = [f.path for f in os.scandir(root) if f.is_dir()]

    # for subfolder in tqdm(subfolders, desc="Processing subfolders"):
    #     src_dir = os.path.join(subfolder, 'src')
    #     tgt_dir = os.path.join(subfolder, 'tgt')

    #     if not os.path.exists(src_dir) or not os.path.exists(tgt_dir):
    #         continue

        src_img_path = glob(os.path.join(src_dir, '*.png'))
        src_cam_path = glob(os.path.join(src_dir, '*.txt'))
        src_img_path.sort()
        src_cam_path.sort()
        src_inps = [(np.array(imageio.imread(img_path)) / 255.) * 2 - 1 for img_path in src_img_path]
        src_ixts, src_exts = [], []
        for cam_path in src_cam_path:
            ixt, ext, _, _ = read_cam_file(cam_path)
            src_ixts.append(ixt.astype(np.float32))
            ext[:3, 3] *= scale_factor
            src_exts.append(ext.astype(np.float32))

        tgt_files = glob(os.path.join(tgt_dir, '*.txt'))

        for tgt_file in tgt_files:
            ret = {}
            tar_ixt, tar_ext, depth_min, depth_max = read_cam_file(tgt_file)
            tar_ext[:3, 3] *= scale_factor
            depth_ranges = [depth_min * scale_factor, depth_max * scale_factor]

            tar_img = np.ones_like(src_inps[0])
            tar_mask = np.ones_like(tar_img[..., 0])
            H, W = tar_img.shape[:2]
            ret['meta'] = {}
            for i in range(cfg.mvsgs.cas_config.num):
                rays, _, _ = mvsgs_utils.build_rays(tar_img, tar_ext, tar_ixt, tar_mask, i, 'test')
                ret[f'rays_{i}'] = torch.from_numpy(rays[None]).cuda()
                ret['meta'][f'h_{i}'] = H
                ret['meta'][f'w_{i}'] = W

            R = np.array(tar_ext[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
            T = np.array(tar_ext[:3, 3], np.float32)
            for i in range(cfg.mvsgs.cas_config.num):
                h, w = H * cfg.mvsgs.cas_config.render_scale[i], W * cfg.mvsgs.cas_config.render_scale[i]
                tar_ixt_ = tar_ixt.copy()
                tar_ixt_[:2, :] *= cfg.mvsgs.cas_config.render_scale[i]
                FovX = data_utils.focal2fov(tar_ixt_[0, 0], w)
                FovY = data_utils.focal2fov(tar_ixt_[1, 1], w)
                projection_matrix = data_utils.getProjectionMatrix(znear=znear, zfar=zfar, K=tar_ixt_, h=h, w=w).transpose(0, 1)
                world_view_transform = torch.tensor(data_utils.getWorld2View2(R, T, np.array(trans), scale)).transpose(0, 1)
                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                camera_center = world_view_transform.inverse()[3, :3]
                novel_view_data = {
                    'FovX': torch.FloatTensor([FovX]).cuda(),
                    'FovY': torch.FloatTensor([FovY]).cuda(),
                    'width': torch.LongTensor([w]).cuda(),
                    'height': torch.LongTensor([h]).cuda(),
                    'world_view_transform': world_view_transform.cuda(),
                    'full_proj_transform': full_proj_transform.cuda(),
                    'camera_center': camera_center.cuda()
                }
                ret[f'novel_view{i}'] = novel_view_data

            src_inps_tensor = np.stack(src_inps)
            src_inps_tensor = torch.from_numpy(src_inps_tensor).cuda().permute(0, 3, 1, 2).type(torch.float32)[None]

            src_exts_tensor = np.stack(src_exts)
            src_exts_tensor = torch.from_numpy(src_exts_tensor).cuda().type(torch.float32)[None]

            src_ixts_tensor = np.stack(src_ixts)
            src_ixts_tensor = torch.from_numpy(src_ixts_tensor).cuda().type(torch.float32)[None]

            tar_ext_tensor = torch.from_numpy(tar_ext).cuda().type(torch.float32)[None]

            tar_ixt_tensor = torch.from_numpy(tar_ixt).cuda().type(torch.float32)[None]

            ret['src_inps'] = src_inps_tensor
            ret['src_exts'] = src_exts_tensor
            ret['src_ixts'] = src_ixts_tensor
            ret['tar_ext'] = tar_ext_tensor
            ret['tar_ixt'] = tar_ixt_tensor
            ret['near_far'] = torch.from_numpy(np.array(depth_ranges)[None].astype(np.float32)).cuda()

            all_batches.append((ret, tgt_file))
    
        return all_batches


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Synthesize novel views from given multi-view images")
    parser.add_argument("--data_dir", type=str, default='demo_data/test')
    parser.add_argument("--save_dir", type=str, default='demo_data/test')
    parser.add_argument("--checkpoint", type=str, default='./trained_model/mvsgs/dtu_pretrain/233.pth')
    args = parser.parse_args()
    config = glob(os.path.join(os.path.join(args.data_dir), '*.yaml'))[0]
    
    scale_factor = read_config(config)
    network = make_network(cfg).cuda()
    pretrained_model = torch.load(args.checkpoint)
    network.load_state_dict(pretrained_model['net'], strict=True)
    network.eval()
    subfolders = [f.path for f in os.scandir(args.data_dir) if f.is_dir()]
    for subfolder in tqdm(subfolders, desc="Processing subfolders"):
        src_dir = os.path.join(subfolder, 'input_images')
        tgt_dir = os.path.join(subfolder, 'tgt')
        all_batches = load_data(src_dir, tgt_dir, scale_factor)
        
        for batch, tgt_file in tqdm(all_batches, desc="Processing batches"):
            H, W = batch['src_inps'].shape[-2:]
            # print("tgt_file----------------------", tgt_file)
            # print("batch", batch)
            with torch.no_grad():
                out = network(batch)
            # print("out", out)
            #--------------------color---------------------------------
            pred = out['rgb_level1']
            # print("pred shape", pred.shape)
            pred = pred.reshape(pred.shape[0], H, W, 3)
            pred = pred.cpu().data.numpy()[0]
            # print("pred", pred)
            
            tgt_filename = os.path.splitext(os.path.basename(tgt_file))[0]
            img_path = os.path.join(tgt_dir, f'{tgt_filename}.png')
            imageio.imwrite(img_path, (pred * 255.).astype(np.uint8))
