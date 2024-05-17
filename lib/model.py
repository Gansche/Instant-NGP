import torch
import torch.nn as nn

class InstantNGP(nn.Module):
    def __init__(self, cfg_enc, cfg_dec) -> None:
        super(InstantNGP, self).__init__()
        
        # self.encoder = HashGridEncoder()
        
        if cfg_dec['arch'] == 'nerf':
            self.decoder = NeuralRadianceField(cfg_dec['nerf'])
        elif cfg_dec['arch'] == 'simple_nerf':
            #! TODO input dim
            self.decoder = SimplifiedNeuralRadianceField(None, None, cfg_dec['simple_nerf'])
        else:
            raise NotImplementedError("[ERROR] No such arch: {}.".format(cfg_dec['arch']))
        
    def forward(self):
        """
        Args:
            input:point?
            output:rgb+sdf?
        """
        pass

class HashGridEncoder(nn.Module):
    def __init__(self) -> None:
        super(HashGridEncoder, self).__init__()
        
    def forward(self, x):
        pass
    
class FrequencyEncoder(nn.Module):
    def __init__(self) -> None:
        super(FrequencyEncoder, self).__init__()
        
    def forward(self, x):
        pass
    
    
#! TODO
class NeuralRadianceField(nn.Module):
    def __init__(self, pos_dim, view_dim, cfg_dec) -> None:
        super(NeuralRadianceField, self).__init__()   
        
#         cfg_sigma = cfg_dec['sigma_net']
#         sigma_model = nn.Sequential(nn.Linear(pos_dim, cfg_sigma['hiden_dim']))
#         for i in range(cfg_sigma['num_layers'] - 1):
#             sigma_model.add_module(nn.ReLU())
#             if i == cfg_sigma['concat'] - 1:
#                 sigma_model.add_module(nn.Linear(pos_dim + cfg_sigma['hiden_dim'], cfg_sigma['hiden_dim']))
#             elif i == cfg_sigma['num_layers'] - 2:
#                 sigma_model.add_module(nn.Linear( cfg_sigma['hiden_dim'], cfg_sigma['hiden_dim'] + 1))
#             else:
#                 sigma_model.add_module(nn.Linear( cfg_sigma['hiden_dim'], cfg_sigma['hiden_dim']))
#         self.sigma_model = sigma_model
        
#         cfg_color = cfg_dec['color_net']
#         color_model = nn.Sequential()
#         for i in range(cfg_color['num_layers']):
#             pass
#         self.color_model = color_model
        
    def forward(self, enc_positoin, enc_view):
        """
        Args:
            enc_position (tensor): encoded position information
            enc_view (tensor): encoded view information
        """
        pass
    
class SimplifiedNeuralRadianceField(nn.Module):
    def __init__(self, pos_dim, view_dim, cfg_dec) -> None:
        super(SimplifiedNeuralRadianceField, self).__init__()
        
        self.pos_dim = pos_dim
        self.view_dim = view_dim
        
        self.cfg_sigma = cfg_dec['sigma_net']
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
        
        self.cfg_color = cfg_dec['color_net']
        color_model = nn.Sequential()
        for i in range(self.cfg_color['num_layers']):
            if i == 0:
                color_model.add_module(nn.Linear(view_dim + self.cfg_sigma['out_dim'] - 1, self.cfg_color['hidden_dim'], bias=False))
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
