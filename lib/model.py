import math
import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import sample_pdf

###############################################################################
""" Instant NGP """

class InstantNGP(nn.Module):
    def __init__(self, cfg, bb) -> None:
        super(InstantNGP, self).__init__()
        
        self.cfg = cfg
        
        self.sampler = Sampler(cfg['sampler'])
        self.importance_sampler = ImportanceSampler(cfg['sampler'])
        self.pos_enc = HashGridEncoder(cfg['hash_grid_enc'], bb)
        # self.dir_enc = FrequencyEncoder(cfg['freq_enc'])
        self.dir_enc = SHEncoder(cfg['sh_enc'])
        self.decoder = SimplifiedNeuralRadianceField(cfg['decoder'])
        self.renderer = Renderer(cfg['renderer'])
        
    def forward(self, rays, perturb=True, importance=True):
        """
        Args:
            input: rays (:, o+d=6)
            output: rgb and sigma (:, 3+1)
        """
        if importance:

            with torch.no_grad():
                pose, dirs, dirs_importance, z_vals = self.sampler(rays, perturb)
                pose_enc = self.pos_enc(pose)
                dirs_enc = self.dir_enc(dirs)
                sigma, color = self.decoder(pose_enc, dirs_enc)
                rgb, weights = self.renderer(sigma ,color, z_vals, rays)
                
            pose, z_vals = self.importance_sampler(rays, z_vals, weights, perturb)
            pose_enc = self.pos_enc(pose)
            dirs_enc_importance = self.dir_enc(dirs_importance)
            sigma, color = self.decoder(pose_enc, dirs_enc_importance)
            rgb, _ = self.renderer(sigma, color, z_vals, rays)
            
        else:
            pose, dirs, _, z_vals = self.sampler(rays, perturb)
            pose_enc = self.pos_enc(pose)
            dirs_enc = self.dir_enc(dirs)
            sigma, color = self.decoder(pose_enc, dirs_enc)
            rgb, _ = self.renderer(sigma ,color, z_vals, rays)

        return rgb
        

###############################################################################
""" Sampler """

class Sampler(nn.Module):
    def __init__(self, cfg) -> None:
        super(Sampler, self).__init__()
        self.near = cfg['near']
        self.far = cfg['far']
        self.num_sample = cfg['num_sample']
        self.num_importance_sample = cfg['num_importance']
        
    def forward(self, rays, perturb):
        #! normalize的
        dirs = rays[..., :3].unsqueeze(-2)
        pts = rays[..., 3:].unsqueeze(-2)
        
        t = torch.linspace(self.near, self.far, self.num_sample).cuda()
        
        repeat_times = rays.shape[:-1] + tuple(1 for _ in range(len(t.shape)))
        t_pt = t.repeat(repeat_times)
        # pdb.set_trace()
        if perturb:
            mid = (t_pt[...,1:] + t_pt[...,:-1]) * 0.5
            upper = torch.cat([mid, t_pt[...,-1:]], -1)
            lower = torch.cat([t_pt[...,:1], mid], -1)
            t_rand = torch.rand(t_pt.shape).cuda()
            t_pt = lower + (upper - lower) * t_rand
        
        t = t_pt

        sampled_pts = (pts + dirs * t.unsqueeze(-1)).squeeze(0)
        sampled_dirs = dirs.repeat(*([1] * (dirs.ndimension() - 2)), self.num_sample, 1)
        sampled_dirs_importance = dirs.repeat(*([1] * (dirs.ndimension() - 2)), self.num_sample + self.num_importance_sample, 1)

        sampled_dirs = sampled_dirs / torch.norm(sampled_dirs, dim=-1, keepdim=True)
        sampled_dirs_importance = sampled_dirs_importance / torch.norm(sampled_dirs_importance, dim=-1, keepdim=True)
        return sampled_pts, sampled_dirs, sampled_dirs_importance, t
    
class ImportanceSampler(nn.Module):
    def __init__(self, cfg) -> None:
        super(ImportanceSampler, self).__init__()
        self.num_sample = cfg['num_importance']

    def forward(self, rays, dists, weights, perturb):
        dirs = rays[..., :3].unsqueeze(-2)
        pts = rays[..., 3:].unsqueeze(-2)
        dists_mid = 0.5 * (dists[...,1:] + dists[...,:-1])
        new_dists = sample_pdf(dists_mid, weights.squeeze(-1)[...,1:-1], self.num_sample, perturb)
        # new_dists = new_dists.detach() ## 避免梯度计算？
        dists, _ = torch.sort(torch.cat([dists, new_dists], -1), -1)

        sampled_pts = (pts + dirs * dists.unsqueeze(-1)).squeeze(0)
        return sampled_pts, dists

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
        self.b = math.exp((math.log(self.N_max) - math.log(self.N_min)) / (self.L - 1))
        self.bb_min = bb[..., :3]
        self.bb_max = bb[..., 3:]

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
        max_vertex = min_vertex + torch.tensor([1.0, 1.0, 1.0]).cuda() * cell
        indices = min_index.unsqueeze(-2) + self.vertices
        return indices, min_vertex, max_vertex
    
    def get_hash_indices(self, xyz):  ## hash function
        assert xyz.shape[-1] == 3
        result = torch.zeros_like(xyz)[..., 0]
        for i in range(xyz.shape[-1]):
            result ^= xyz[..., i] * self.pi[i]
        return torch.tensor((1<<19)-1).to(result.device) & result

    def get_interpolation(self, xyz, min_v, max_v, emb_indices):
        d = (xyz - min_v) / (max_v - min_v)
        c00 = emb_indices[..., 0, :] * (1 - d[..., 0][..., None]) + emb_indices[..., 4, :] * d[..., 0][..., None]
        c01 = emb_indices[..., 1, :] * (1 - d[..., 0][..., None]) + emb_indices[..., 5, :] * d[..., 0][..., None]
        c10 = emb_indices[..., 2, :] * (1 - d[..., 0][..., None]) + emb_indices[..., 6, :] * d[..., 0][..., None]
        c11 = emb_indices[..., 3, :] * (1 - d[..., 0][..., None]) + emb_indices[..., 7, :] * d[..., 0][..., None]
        c0 = c00 * (1 - d[..., 1][..., None]) + c10 * d[..., 1][..., None]
        c1 = c01 * (1 - d[..., 1][..., None]) + c11 * d[..., 1][..., None]
        c = c0 * (1 - d[..., 2][..., None]) + c1 * d[..., 2][..., None]
        return c
    
    def forward(self, xyz):
        """
        Args:
            xyz (tensor): input position
        """
        assert xyz.shape[-1] == 3
        embedded = []
        for i in range(self.L):
            indices, min_vertex, max_vertex = self.get_indices_vertices(i, xyz)
            hash_indices = self.get_hash_indices(indices)
            embedded_indices = self.hash_grids[i](hash_indices)
            embedded_i = self.get_interpolation(
                xyz,                    # [1024, 64, 3]
                min_vertex, max_vertex, # [1024, 64, 3]
                embedded_indices        # [1024, 64, 8, 2]
            )
            embedded.append(embedded_i)
        return torch.cat(embedded, dim=-1)
        
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
        return torch.cat(enc_x, dim=-1)
    
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
                sigma_model.append(nn.Linear(self.pos_dim, self.cfg_sigma['hidden_dim'], bias=False))
                sigma_model.append(nn.ReLU())
            elif i == self.cfg_sigma['num_layers'] - 1:
                sigma_model.append(nn.Linear(self.cfg_sigma['hidden_dim'], self.cfg_sigma['out_dim'], bias=False))
            else:
                sigma_model.append(nn.Linear(self.cfg_sigma['hidden_dim'], self.cfg_sigma['hidden_dim'], bias=False))
                sigma_model.append(nn.ReLU())
        self.sigma_model = sigma_model
        
        self.cfg_color = cfg['color_net']
        color_model = nn.Sequential()
        for i in range(self.cfg_color['num_layers']):
            if i == 0:
                color_model.append(nn.Linear(self.view_dim + self.cfg_sigma['out_dim'] - 1, self.cfg_color['hidden_dim'], bias=False))
                color_model.append(nn.ReLU())
            elif i == self.cfg_color['num_layers'] - 1:
                color_model.append(nn.Linear(self.cfg_color['hidden_dim'], self.cfg_color['out_dim'], bias=False))
                color_model.append(nn.Sigmoid())
            else:
                color_model.append(nn.Linear(self.cfg_color['hidden_dim'], self.cfg_color['hidden_dim'], bias=False))
                color_model.append(nn.ReLU())
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
        color = self.color_model(torch.cat((view_enc, latent_vc), dim=-1))

        return sigma, color

###############################################################################
""" Renderer """

class Renderer(nn.Module):
    def __init__(self, cfg) -> None:
        super(Renderer, self).__init__()
        
    def forward(self, sigma, color, z_vals, rays):
        dists = torch.cat(
            [
                z_vals[...,1:] - z_vals[...,:-1], 
                torch.tensor([1e10]).expand(z_vals[...,:1].shape).cuda()
            ], dim=-1
        )
        dirs = rays[..., :3].unsqueeze(-2)
        dists = dists * torch.norm(dirs, dim=-1)
        alpha = 1.0 - torch.exp((-1) * sigma * dists[..., None])
        temp = torch.cat((torch.ones_like(alpha[..., 0, :][..., None], dtype=torch.float32), 1.0 - alpha + 1e-10), dim=-2)
        T = torch.cumprod(temp[..., :-1, :], dim=-2)
        weights = alpha * T
        rgb = torch.sum(weights * color, dim=-2)
        
        return rgb, weights
