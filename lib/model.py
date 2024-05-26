import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import pdb

###############################################################################
""" Instant NGP """

class InstantNGP(nn.Module):
    def __init__(self, cfg, bb) -> None:
        super(InstantNGP, self).__init__()
        
        self.cfg = cfg
        
        self.sampler = Sampler(cfg['sampler'])
        self.pos_enc = HashGridEncoder(cfg['hash_grid_enc'], bb)
        # self.dir_enc = FrequencyEncoder(cfg['freq_enc'])
        self.dir_enc = SHEncoder(cfg['sh_enc'])
        # self.decoder = SimplifiedNeuralRadianceField(cfg['decoder'])
        self.renderer = Renderer(cfg['renderer'])
        
    def forward(self, rays):
        """
        Args:
            input: rays (:, o+d=6)
            output: rgb and sigma (:, 3+1)
        """
        # sampling
        pts, dirs, dists = self.sampler(rays)
        
        # encoding
        enc_pts = self.pos_enc(pts)
        enc_dirs = self.dir_enc(dirs)
        
        # nerfing
        
        # rendering
        
        return None
        

###############################################################################
""" Sampler """

class Sampler(nn.Module):
    def __init__(self, cfg) -> None:
        super(Sampler, self).__init__()
        self.near = cfg['near']
        self.far = cfg['far']
        self.num_sample = cfg['num_sample']
        
    def forward(self, rays):
        #! 这里的rays方向是没有normalize的，先跟HashNeRF对齐写出来再说
        dirs = rays[..., :3].unsqueeze(-2)
        pts = rays[..., 3:].unsqueeze(-2)
        
        t = torch.linspace(self.near, self.far, self.num_sample).cuda()
        
        # perturb
        repeat_times = rays.shape[:-1] + tuple(1 for _ in range(len(t.shape)))
        t_pt = t.repeat(repeat_times)
        mid = (t_pt[...,1:] +t_pt[...,:-1]) * 0.5
        upper = torch.cat([mid, t_pt[...,-1:]], -1)
        lower = torch.cat([t_pt[...,:1], mid], -1)
        t_rand = torch.rand(t_pt.shape).cuda()
        t_pt = lower + (upper - lower) * t_rand
        dists = torch.cat(
            [
                t_pt[...,1:] - t_pt[...,:-1], 
                torch.tensor([1e10]).expand(t_pt[...,:1].shape).cuda()
            ], dim=-1
        )

        sampled_pts = (pts + dirs * t.unsqueeze(-1)).squeeze(0)
        sampled_dirs = dirs.repeat(*([1] * (dirs.ndimension() - 2)), self.num_sample, 1)
        
        return sampled_pts, sampled_dirs, dists

###############################################################################
""" Encoder """

class HashGridEncoder(nn.Module):
    def __init__(self, cfg, bb) -> None:
        super(HashGridEncoder, self).__init__()
        self.L = cfg['L']
        self.T = int(math.pow(2, cfg['log2T']))
        self.F = int(cfg['F'])
        self.N_min = cfg['N_min']
        self.N_max = cfg['N_max']
        self.b = math.exp(math.log(self.N_max) - math.log(self.N_min) / (self.L - 1))
        self.bb_min, self.bb_max = bb

        hash_grids = []
        resolutions = []
        for i in range(self.L):
            embedding = nn.Embedding(self.T, self.F)
            nn.init.uniform_(embedding.weight, a=-0.0001, b=0.0001)
            hash_grids.append(embedding)
            resolutions.append(math.floor(self.N_min * math.pow(self.b, i)))
        self.hash_grids = nn.ModuleList(hash_grids)
        self.resolutions = resolutions
        
        self.vertices = torch.tensor(
            [
                [0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], 
                [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]
            ]
        ).cuda()
        self.pi = [1, 2654435761, 805459861]
    
    def get_indices_vertices(self, i, xyz):
        cell = (self.bb_max - self.bb_min) / self.resolutions[i]
        min_index = torch.floor((xyz - self.bb_min) / cell).int()
        min_vertex = min_index * cell + self.bb_min
        max_vertex = min_vertex + torch.tensor([1, 1, 1]).cuda() * cell
        indices = min_index.unsqueeze(-2) + self.vertices
        return indices, min_vertex, max_vertex
    
    def get_hash_indices(self, xyz):
        assert xyz.shape[-1] == 3
        result = torch.zeros_like(xyz)[..., 0]
        for i in range(xyz.shape[-1]):
            result ^= xyz[..., i] * self.pi[i]
        return result % self.T

    def get_interpolation(self, xyz, min_v, max_v, emb_indices):
        """
            xyz,                    # [1024, 64, 3]
            min_vertex, max_vertex, # [1024, 64, 8, 3]
            embedded_indices        # [1024, 64, 8, 2]
        """
        d = (xyz - min_v) / (max_v - min_v)
        pdb.set_trace()
        c00 = emb_indices[..., 0] * (1 - d[:, 0][:, None]) + emb_indices[:,4] * d[:, 0][:, None]
        c01 = emb_indices[..., 1] * (1 - d[:, 0][:, None]) + emb_indices[:,5] * d[:, 0][:, None]
        c10 = emb_indices[:, 2] * (1 - d[:, 0][:, None]) + emb_indices[:,6] * d[:, 0][:, None]
        c11 = emb_indices[:, 3] * (1 - d[:, 0][:, None]) + emb_indices[:,7] * d[:, 0][:, None]
        
        # c0 = c00 * (1 - d[:, 1][:, None]) + c10 * d[:, 1][:, None]
        # c1 = c01 * (1 - d[:, 1][:, None]) + c11 * d[:, 1][:, None]
        
        # c = c0 * (1 - d[:, 2][:, None]) + c1 * d[:, 2][:, None]
        
        return None
    
    def forward(self, xyz):
        """
        Args:
            x (tensor): input position
        """
        assert xyz.shape[-1] == 3
        embedded = []
        for i in range(self.L):
            indices, min_vertex, max_vertex = self.get_indices_vertices(i, xyz)
            hash_indices = self.get_hash_indices(indices)
            embedded_indices = self.hash_grids[i](hash_indices)
            embedded_i = self.get_interpolation(
                xyz,                    # [1024, 64, 3]
                min_vertex, max_vertex, # [1024, 64, 8, 3]
                embedded_indices        # [1024, 64, 8, 2]
            )
            pdb.set_trace()
            embedded.append(embedded_i)
        enc_pts = torch.cat(embedded, dim=-1)
        return enc_pts
        
class FrequencyEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super(FrequencyEncoder, self).__init__()
        self.L = cfg['L']
        self.funcs = [torch.sin, torch.cos]
        
    def forward(self, x):
        """
        Method:
            enc(p) = (sin(2^0p), cos(2^0p) ... sin(2^(2L-1)p), cos(2^(2L-1)p))
        """
        enc_x = []
        for i in range(self.L * 2):
            for func in self.funcs:
                enc_x.append(func(x * (2 ** i)))
        enc_x = torch.cat(enc_x, dim=-1)
        return enc_x
    
class SHEncoder(nn.Module):
    def __init__(self, cfg):
    
        super().__init__()

        self.input_dim = cfg['input_dim']
        self.degree = cfg['degree']

        assert self.input_dim == 3
        assert self.degree >= 1 and self.degree <= 5

        self.out_dim = self.degree ** 2

        self.C0 = 0.28209479177387814
        self.C1 = 0.4886025119029199
        self.C2 = [
            1.0925484305920792,
            -1.0925484305920792,
            0.31539156525252005,
            -1.0925484305920792,
            0.5462742152960396
        ]
        self.C3 = [
            -0.5900435899266435,
            2.890611442640554,
            -0.4570457994644658,
            0.3731763325901154,
            -0.4570457994644658,
            1.445305721320277,
            -0.5900435899266435
        ]
        self.C4 = [
            2.5033429417967046,
            -1.7701307697799304,
            0.9461746957575601,
            -0.6690465435572892,
            0.10578554691520431,
            -0.6690465435572892,
            0.47308734787878004,
            -1.7701307697799304,
            0.6258357354491761
        ]

    def forward(self, input, **kwargs):

        result = torch.empty((*input.shape[:-1], self.out_dim), dtype=input.dtype, device=input.device)
        x, y, z = input.unbind(-1)

        result[..., 0] = self.C0
        if self.degree > 1:
            result[..., 1] = -self.C1 * y
            result[..., 2] = self.C1 * z
            result[..., 3] = -self.C1 * x
            if self.degree > 2:
                xx, yy, zz = x * x, y * y, z * z
                xy, yz, xz = x * y, y * z, x * z
                result[..., 4] = self.C2[0] * xy
                result[..., 5] = self.C2[1] * yz
                result[..., 6] = self.C2[2] * (2.0 * zz - xx - yy)
                #result[..., 6] = self.C2[2] * (3.0 * zz - 1) # xx + yy + zz == 1, but this will lead to different backward gradients, interesting...
                result[..., 7] = self.C2[3] * xz
                result[..., 8] = self.C2[4] * (xx - yy)
                if self.degree > 3:
                    result[..., 9] = self.C3[0] * y * (3 * xx - yy)
                    result[..., 10] = self.C3[1] * xy * z
                    result[..., 11] = self.C3[2] * y * (4 * zz - xx - yy)
                    result[..., 12] = self.C3[3] * z * (2 * zz - 3 * xx - 3 * yy)
                    result[..., 13] = self.C3[4] * x * (4 * zz - xx - yy)
                    result[..., 14] = self.C3[5] * z * (xx - yy)
                    result[..., 15] = self.C3[6] * x * (xx - 3 * yy)
                    if self.degree > 4:
                        result[..., 16] = self.C4[0] * xy * (xx - yy)
                        result[..., 17] = self.C4[1] * yz * (3 * xx - yy)
                        result[..., 18] = self.C4[2] * xy * (7 * zz - 1)
                        result[..., 19] = self.C4[3] * yz * (7 * zz - 3)
                        result[..., 20] = self.C4[4] * (zz * (35 * zz - 30) + 3)
                        result[..., 21] = self.C4[5] * xz * (7 * zz - 3)
                        result[..., 22] = self.C4[6] * (xx - yy) * (7 * zz - 1)
                        result[..., 23] = self.C4[7] * xz * (xx - 3 * yy)
                        result[..., 24] = self.C4[8] * (xx * (xx - 3 * yy) - yy * (3 * xx - yy))

        return result
    
###############################################################################
""" NeRF model """
class SimplifiedNeuralRadianceField(nn.Module):
    def __init__(self,cfg) -> None:
        super(SimplifiedNeuralRadianceField, self).__init__()
        
        self.pos_dim = cfg['pos_dim']
        self.view_dim = cfg['view_dim']
        
        self.cfg_sigma = cfg['sigma_net']
        sigma_model = nn.Sequential()
        for i in range(self.cfg_sigma['num_layers']):
            if i == 0:
                sigma_model.add_module(nn.Linear(self.pos_dim, self.cfg_sigma['hidden_dim'], bias=False))
                sigma_model.add_module(nn.ReLU())
            elif i == self.cfg_sigma['num_layers'] - 1:
                sigma_model.add_module(nn.Linear(self.cfg_sigma['hidden_dim'], self.cfg_sigma['out_dim'], bias=False))
            else:
                sigma_model.add_module(nn.Linear(self.cfg_sigma['hidden_dim'], self.cfg_sigma['hidden_dim'], bias=False))
                sigma_model.add_module(nn.ReLU())
        self.sigma_model = sigma_model
        
        self.cfg_color = cfg['color_net']
        color_model = nn.Sequential()
        for i in range(self.cfg_color['num_layers']):
            if i == 0:
                color_model.add_module(nn.Linear(self.view_dim + self.cfg_sigma['out_dim'] - 1, self.cfg_color['hidden_dim'], bias=False))
                color_model.add_module(nn.ReLU())
            elif i == self.cfg_color['num_layers'] - 1:
                color_model.add_module(nn.Linear(self.cfg_color['hidden_dim'], self.cfg_color['out_dim'], bias=False))
                color_model.add_module(nn.Sigmoid())
            else:
                color_model.add_module(nn.Linear(self.cfg_color['hidden_dim'], self.cfg_color['hidden_dim'], bias=False))
                color_model.add_module(nn.ReLU())
        self.color_model = color_model
        
    def forward(self, pos_enc, view_enc):
        """
        Args:
            enc_position (tensor): encoded position information (..., ?)
            enc_view (tensor): encoded view information (..., ?)
            
            sigma (tensor): sigma (..., 1)
            color (tensor): color (..., 3)
        """
        output = self.sigma_model(pos_enc)
        sigma = F.relu(output[..., 0].unsqueeze(-1))
        latent_vc = output[..., 1:]
        
        color = self.color_model(torch.cat(view_enc, latent_vc, dim=-1))
        return sigma, color

###############################################################################
""" Renderer """

class Renderer(nn.Module):
    def __init__(self, cfg) -> None:
        super(Renderer, self).__init__()
        
    def forward(self, sigma, color, dists):
        alpha = 1 - torch.exp(sigma * dists)
        weight = None
        
        rgb = None
        return rgb
