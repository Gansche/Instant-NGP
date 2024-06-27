import numpy as np
from tqdm import tqdm

import torch
import imageio
import argparse

from lib.model import InstantNGP
from lib.dataset import make_dataset

torch.manual_seed(0)
torch.cuda.manual_seed(0)

###########################################################################
""" ckpt """
parser = argparse.ArgumentParser(description="Training stylizer")
parser.add_argument('-d', '--data_path', type=str, default='hotdog', help='config file path')
args = parser.parse_args()
data_name = args.data_path
ckpt = torch.load(f'ckpt/ckpt_{data_name}_09999_epoch.pth')
config = ckpt['config']
net = ckpt['net']

###########################################################################
""" re-create boundingbox. fuck """
dataset =  make_dataset(config['data'], 'data/' + data_name)
bb = dataset.create_bounding_box().cuda()

###########################################################################
""" model """
model = InstantNGP(config, bb)
model.load_state_dict(net)
model.cuda()

###########################################################################
""" create pose """
trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()
def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w
render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180,180,40+1)[:-1]], 0)

###########################################################################
""" render """
to8b = lambda x : (255*np.clip(x,0,1)).astype(np.uint8)

def render():
    pass

def create_rays(pose):
    c2w = pose
    x, y = torch.meshgrid(
        torch.arange(dataset.W, dtype=torch.float32), 
        torch.arange(dataset.H, dtype=torch.float32), 
        indexing='xy'
    )
    dirs = torch.stack([(x - dataset.W * 0.5) / dataset.focal, -(y - dataset.H * 0.5) / dataset.focal, -torch.ones_like(x)], -1)
    dirs_world = dirs @ c2w[:3, :3].T
    pts =  c2w[:3, -1].expand(dataset.H, dataset.W, 3)
    rays = torch.cat([dirs_world, pts], dim=-1)
    return rays

rgbs = []
for idx in tqdm(range(render_poses.shape[0])):
    pose = render_poses[idx]
    rays = create_rays(pose)
    
    batch_size = 4096
    rays_reshape = rays.reshape(dataset.H * dataset.W, 6)
    renderings = []
    for i in range(dataset.H * dataset.W // batch_size + 1):
        with torch.no_grad():
            ret = model(rays_reshape[i * batch_size : (i + 1) * batch_size].cuda(), False, True)
        renderings.append(ret.cpu())
    image = torch.cat(renderings, dim=0).reshape(dataset.H, dataset.W, 3)
    pred_image = to8b(image.numpy())
    rgbs.append(pred_image)
rgbs = np.stack(rgbs, 0)
imageio.mimwrite(f'{data_name}_rgb.mp4', rgbs, fps=30, quality=8)