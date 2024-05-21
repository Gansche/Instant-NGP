import math

import torch
import torch.nn as nn

import pdb

###############################################################################
""" Instant NGP """

class InstantNGP(nn.Module):
    def __init__(self, cfg) -> None:
        super(InstantNGP, self).__init__()
        
        self.cfg = cfg
        
        self.sampler = Sampler(cfg['sampler'])
        self.pos_enc = HashGridEncoder(cfg['hash_grid_enc'])
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
        pts, dirs = self.sampler(rays)
        enc_dirs = self.dir_enc(dirs)
        pdb.set_trace()

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
        #! 等距采样不perturb
        dirs = rays[..., :3].unsqueeze(-2)
        pts = rays[..., 3:].unsqueeze(-2)
        
        t = torch.linspace(self.near, self.far, self.num_sample).cuda()
        t = t.view(*([1] * (len(pts.shape) - 1)), self.num_sample, 1)
        sampled_pts = (pts + t * dirs).squeeze(0)
        sampled_dirs = dirs.repeat(*([1] * (dirs.ndimension() - 2)), self.num_sample, 1)
        
        return sampled_pts, sampled_dirs
        

###############################################################################
""" Encoder """

class HashGridEncoder(nn.Module):
    def __init__(self, cfg) -> None:
        super(HashGridEncoder, self).__init__()
        self.L = cfg['L']
        self.T = int(math.pow(2, cfg['log2T']))
        self.F = int(cfg['F'])
        self.N_min = cfg['N_min']
        self.N_max = cfg['N_max']
        self.b = math.exp(math.log(self.N_max) - math.log(self.N_min) / (self.L - 1))

        hash_grids = []
        resolutions = []
        for i in range(self.L):
            hash_grids.append(nn.Embedding(self.T, self.F))
            resolutions.append(math.floor(self.N_min * math.pow(self.b, i)))
        self.hash_grids = nn.ModuleList(hash_grids)
        self.resolutions = resolutions
        
    def forward(self, x):
        """
        Args:
            x (tensor): input position
        """
        assert x.shape[-1] == 3
        
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
                #! HashNeRF has no sigmoid layer but NeRF does, I wonder what will happen if add a sigmoid layer...
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
        sigma = output[..., 0].unsqueeze(-1)
        latent_vc = output[..., 1:]
        
        color = self.color_model(torch.cat(view_enc, latent_vc, dim=-1))
        return sigma, color

###############################################################################
""" Renderer """

class Renderer(nn.Module):
    def __init__(self, cfg) -> None:
        super(Renderer, self).__init__()
        
    def forward(self, x):
        rgb = None
        return rgb
