import numpy as np
import os
from lib.datasets import mvsgs_utils
from lib.config import cfg
import imageio
import cv2
import random
from lib.utils import data_utils
import torch
from lib.utils.video_utils import *

if cfg.fix_random:
    random.seed(0)
    np.random.seed(0)
def rgb_to_depth_map(rgb_image, far_plane=1000.0):
    """Chuyển đổi ảnh RGB thành depth map."""
    B = rgb_image[:, :, 0].astype(np.float32)
    G = rgb_image[:, :, 1].astype(np.float32)
    R = rgb_image[:, :, 2].astype(np.float32)

    depth_map = R + G * 256 + B * 256 * 256
    depth_map = depth_map / (256 * 256 * 256 - 1)  # Chuẩn hóa giá trị
    depth_map *= far_plane
    return depth_map
class Dataset:
    def __init__(self, **kwargs):
        super(Dataset, self).__init__()
        self.data_root = os.path.join(cfg.workspace, kwargs['data_root'])
        print("data_root", self.data_root)
        self.split = kwargs['split']
        print(self.split)
        if 'scene' in kwargs:
            self.scenes = [kwargs['scene']]
        else:
            self.scenes = []
        self.build_metas(kwargs['ann_file'])
        self.depth_ranges = [100., 1000.]
        # self.depth_ranges = [10., 1000.]
        self.zfar = 100.0
        self.znear = 0.01
        self.trans = [0.0, 0.0, 0.0]
        self.scale = 1.0

    def build_metas(self, ann_file):
        scenes = [line.strip() for line in open(ann_file).readlines()]
        dtu_pairs = torch.load('data/mvsgs/pairs.th')

        self.scene_infos = {}
        self.metas = []
        if len(self.scenes) != 0:
            scenes = self.scenes
            
        count = 0
        for scene in scenes:
                
            # if count == 2:
            #     exit(0)
            # count += 1
            # scene_info = {'ixts': [], 'exts': [], 'dpt_paths': [], 'img_paths': []}
            # scene_info = {'ixts': [], 'exts': [], 'img_paths': []}
            print("scene", os.path.join(self.data_root,self.split, f'{scene}'))
            view = int(scene.split("_")[-1])
                # j = 0
            for j in range(4):
                scene_info = {'ixts': [], 'exts': [], 'dpt_paths': [], 'img_paths': []}
                for i in range(6):
                    cam_path = os.path.join(self.data_root, f'camera/input_camera_{i+1+(view-1)*20 + j*5}.txt')
                    ixt, ext, _ = data_utils.read_cam_file(cam_path)
                    ext[:3, 3] = ext[:3, 3]
                    ixt[:2] = ixt[:2] * 4
                    dpt_path = os.path.join(self.data_root,self.split, f'{scene}/input_images_raw/input_camera_{i+1+(view-1)*20 + j*5}.png')
                    # print("dpt_path", dpt_path)
                    img_path = os.path.join(self.data_root,self.split, f'{scene}/input_images/input_camera_{i+1+(view-1)*20 + j*5}.png')
                    # print("img_path", img_path)
                    # exit(0)
                    scene_info['ixts'].append(ixt.astype(np.float32))
                    scene_info['exts'].append(ext.astype(np.float32))
                    scene_info['dpt_paths'].append(dpt_path)
                    scene_info['img_paths'].append(img_path)
                

                if self.split == 'train' and len(self.scenes) != 1:
                    # train_ids = np.arange(49).tolist()
                    # test_ids = np.arange(49).tolist()
                    train_ids = np.arange(21).tolist()
                    test_ids = np.arange(21).tolist()
                elif self.split == 'train' and len(self.scenes) == 1:
                    train_ids = dtu_pairs['dtu_train']
                    test_ids = dtu_pairs['dtu_train']
                else:
                    train_ids = dtu_pairs['dtu_train']
                    test_ids = dtu_pairs['dtu_val']
                train_ids = np.arange(5).tolist()
                # train_ids = [0, 5]
                test_ids = np.random.randint(1, 4, size=1)
                scene_info.update({'train_ids': train_ids, 'test_ids': test_ids})
                new_scene = scene + f"_part_{j}"
                self.scene_infos[new_scene] = scene_info
                # print("scene_infos", self.scene_infos)
                # print("\n")
                # print("----------------------------------------------------------------------------------")
                # print("self.scene_infos", self.scene_infos)
                # print(np.linalg.inv(scene_info['exts'][0])[:3, 3])
                # exit(0)
                cam_points = np.array([np.linalg.inv(scene_info['exts'][i])[:3, 3] for i in train_ids])     
                for tar_view in test_ids:
                    threshold_random = np.random.rand()
                    if threshold_random > 0.5:
                        cam_point = np.linalg.inv(scene_info['exts'][tar_view])[:3, 3]
                        distance = np.linalg.norm(cam_points - cam_point[None], axis=-1)
                        argsorts = distance.argsort()
                        argsorts = argsorts[1:] if tar_view in train_ids else argsorts
                        # input_views_num = cfg.mvsgs.train_input_views[0] if self.split == 'train' else cfg.mvsgs.test_input_views
                        input_views_num = cfg.mvsgs.train_input_views[0] if self.split == 'train' else cfg.mvsgs.test_input_views
                        src_views = [train_ids[i] for i in argsorts[:input_views_num]]
                        # print(src_views)
                    else:
                        src_views = [0,5]
                    
                    self.metas += [(new_scene, tar_view, src_views)]
                    # print("threshold - self.metas",threshold_random,  (new_scene, tar_view, src_views))
                
            # exit(0)
    def __getitem__(self, index_meta):
        index, input_views_num = index_meta
        scene, tar_view, src_views = self.metas[index]

  

        if self.split == 'train':
            if random.random() < 0.1:
                src_views = src_views + [tar_view]
            src_views = random.sample(src_views[:input_views_num+1], input_views_num)

            # src_views = random.sample(src_views[:input_views_num], input_views_num)
        src_views.sort()
        scene_info = self.scene_infos[scene]

        tar_img = np.array(imageio.imread(scene_info['img_paths'][tar_view])) / 255.
        H, W = tar_img.shape[:2]
        # print("H, W", H, W)
        # exit(0)
        tar_ext, tar_ixt = scene_info['exts'][tar_view], scene_info['ixts'][tar_view]
        if self.split != 'train': # only used for evaluation
            # tar_dpt = data_utils.read_pfm(scene_info['dpt_paths'][tar_view])[0].astype(np.float32)
            # tar_dpt = cv2.resize(tar_dpt, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_NEAREST)
            # tar_dpt = tar_dpt[44:556, 80:720]
            # tar_mask = (tar_dpt > 0.).astype(np.uint8)
            rgb_image = cv2.imread(scene_info['dpt_paths'][tar_view], cv2.IMREAD_COLOR)
    
            if rgb_image is None:
                raise FileNotFoundError(f"Không tìm thấy ảnh tại {scene_info['dpt_paths'][tar_view]}")

            # Chuyển đổi ảnh RGB thành depth map
            tar_dpt = rgb_to_depth_map(rgb_image)

            # Thay đổi kích thước ảnh
            # tar_dpt = cv2.resize(tar_dpt, None, fx=1, fy=1, interpolation=cv2.INTER_NEAREST)
            # print("tar_dpt shape", tar_dpt.shape)
            # Cắt vùng quan tâm
            # tar_dpt = tar_dpt[44:556, 80:720]

            # Tạo mặt nạ
            tar_mask = (tar_dpt > 0.).astype(np.uint8)
        else:
            tar_dpt = np.ones_like(tar_img)
            tar_mask = np.ones_like(tar_img)

        # print("src_views", src_views)
        src_inps, src_exts, src_ixts = self.read_src(scene_info, src_views)
        
        # exit(0)
        ret = {'src_inps': src_inps,
               'src_exts': src_exts,
               'src_ixts': src_ixts}
        ret.update({'tar_ext': tar_ext,
                    'tar_ixt': tar_ixt})
        # if self.split != 'train':
        ret.update({'tar_img': tar_img,
                    'tar_dpt': tar_dpt,
                    'tar_mask': tar_mask})
        ret.update({'near_far': np.array(self.depth_ranges).astype(np.float32)})
        ret.update({'meta': {'scene': scene, 'tar_view': tar_view, 'frame_id': 0}})

        for i in range(cfg.mvsgs.cas_config.num):
            rays, rgb, msk = mvsgs_utils.build_rays(tar_img, tar_ext, tar_ixt, tar_mask, i, self.split)
            s = cfg.mvsgs.cas_config.volume_scale[i]
            if self.split != 'train': # evaluation
                tar_dpt_i = cv2.resize(tar_dpt, None, fx=s, fy=s, interpolation=cv2.INTER_NEAREST)
                ret.update({f'tar_dpt_{i}': tar_dpt_i.astype(np.float32)})
            ret.update({f'rays_{i}': rays, f'rgb_{i}': rgb.astype(np.float32), f'msk_{i}': msk})
            ret['meta'].update({f'h_{i}': H, f'w_{i}': W})
            
        R = np.array(tar_ext[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
        T = np.array(tar_ext[:3, 3], np.float32)
        for i in range(cfg.mvsgs.cas_config.num):
            h, w = H*cfg.mvsgs.cas_config.render_scale[i], W*cfg.mvsgs.cas_config.render_scale[i]
            tar_ixt_ = tar_ixt.copy()
            tar_ixt_[:2,:] *= cfg.mvsgs.cas_config.render_scale[i]
            FovX = data_utils.focal2fov(tar_ixt_[0, 0], w)
            FovY = data_utils.focal2fov(tar_ixt_[1, 1], h)
            projection_matrix = data_utils.getProjectionMatrix(znear=self.znear, zfar=self.zfar, K=tar_ixt_, h=h, w=w).transpose(0, 1)
            world_view_transform = torch.tensor(data_utils.getWorld2View2(R, T, np.array(self.trans), self.scale)).transpose(0, 1)
            full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
            camera_center = world_view_transform.inverse()[3, :3]
            novel_view_data = {
                'FovX':  torch.FloatTensor([FovX]),
                'FovY':  torch.FloatTensor([FovY]),
                'width': w,
                'height': h,
                'world_view_transform': world_view_transform,
                'full_proj_transform': full_proj_transform,
                'camera_center': camera_center
            }
            ret[f'novel_view{i}'] = novel_view_data
        
        if cfg.save_video:
            rendering_video_meta = []
            render_path_mode = 'interpolate'            
            poses_paths = self.get_video_rendering_path(ref_poses=src_exts, mode=render_path_mode, near_far=None, train_c2w_all=None, n_frames=60)
            for pose in poses_paths[0]:
                R = np.array(pose[:3, :3], np.float32).reshape(3, 3).transpose(1, 0)
                T = np.array(pose[:3, 3], np.float32)
                FovX = data_utils.focal2fov(tar_ixt[0, 0], W)
                FovY = data_utils.focal2fov(tar_ixt[1, 1], H)
                projection_matrix = data_utils.getProjectionMatrix(znear=self.znear, zfar=self.zfar, K=tar_ixt, h=H, w=W).transpose(0, 1)
                world_view_transform = torch.tensor(data_utils.getWorld2View2(R, T, np.array(self.trans), self.scale)).transpose(0, 1)
                full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
                camera_center = world_view_transform.inverse()[3, :3]
                rendering_meta = {
                    'FovX':  torch.FloatTensor([FovX]),
                    'FovY':  torch.FloatTensor([FovY]),
                    'width': W,
                    'height': H,
                    'world_view_transform': world_view_transform,
                    'full_proj_transform': full_proj_transform,
                    'camera_center': camera_center,
                    'tar_ext': pose
                }
                for i in range(cfg.mvsgs.cas_config.num):
                    tar_ext[:3] = pose
                    rays, _, _ = mvsgs_utils.build_rays(tar_img, tar_ext, tar_ixt, tar_mask, i, self.split)
                    rendering_meta.update({f'rays_{i}': rays})
                rendering_video_meta.append(rendering_meta)
            ret['rendering_video_meta'] = rendering_video_meta
        return ret
    
    def get_video_rendering_path(self, ref_poses, mode, near_far, train_c2w_all, n_frames=60, batch=None):
        # loop over batch
        poses_paths = []
        ref_poses = ref_poses[None]
        for batch_idx, cur_src_poses in enumerate(ref_poses):
            if mode == 'interpolate':
                # convert to c2ws
                pose_square = torch.eye(4).unsqueeze(0).repeat(cur_src_poses.shape[0], 1, 1)
                cur_src_poses = torch.from_numpy(cur_src_poses)
                pose_square[:, :3, :] = cur_src_poses[:,:3]
                cur_c2ws = pose_square.double().inverse()[:, :3, :].to(torch.float32).cpu().detach().numpy()
                cur_path = get_interpolate_render_path(cur_c2ws, n_frames)
            elif mode == 'spiral':
                cur_c2ws_all = train_c2w_all
                cur_near_far = near_far.tolist()
                rads_scale = 0.3
                cur_path = get_spiral_render_path(cur_c2ws_all, cur_near_far, rads_scale=rads_scale, N_views=n_frames)
            else:
                raise Exception(f'Unknown video rendering path mode {mode}')

            # convert back to extrinsics tensor
            cur_w2cs = torch.tensor(cur_path).inverse()[:, :3].to(torch.float32)
            poses_paths.append(cur_w2cs)

        poses_paths = torch.stack(poses_paths, dim=0)
        return poses_paths

    def read_src(self, scene_info, src_views):
       
        inps, exts, ixts = [], [], []
        # print("src_views\n", src_views)
        # print("scene_info['img_paths']\n", scene_info['img_paths']) 
        # print("scene_info['exts']\n", scene_info['exts']) 
        for src_view in src_views:
            # print("scene_info['img_paths']", scene_info['img_paths'][src_view])  
            inps.append((np.array(imageio.imread(scene_info['img_paths'][src_view])) / 255.) * 2. - 1.)
            exts.append(scene_info['exts'][src_view])
            ixts.append(scene_info['ixts'][src_view])
        # print("exts\n", exts)
        # print("--------------------------------------------------------------------------")
        return np.stack(inps).transpose((0, 3, 1, 2)).astype(np.float32), np.stack(exts), np.stack(ixts)

    def __len__(self):
        return len(self.metas)

