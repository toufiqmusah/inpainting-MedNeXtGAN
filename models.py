"""models.py"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from nnunet_mednext import create_mednext_v1
from generative.networks.nets import PatchDiscriminator



class MedNextGenerator3D(nn.Module):
    def __init__(self, input_channels=1, output_channels=1):
        super(MedNextGenerator3D, self).__init__()

        self.model = create_mednext_v1(
            num_input_channels=input_channels,
            num_classes=output_channels,
            model_id='M',              
            kernel_size=3,
            deep_supervision=False,
        )

        self._replace_norm_with_instancenorm()        
        self.final_activation = nn.Tanh()
    
    def _replace_norm_with_instancenorm(self):
        for name, module in self.model.named_modules():

            if isinstance(module, nn.GroupNorm) or isinstance(module, nn.LayerNorm):
                parent_name = '.'.join(name.split('.')[:-1])
                attr_name = name.split('.')[-1]
                
                parent = self.model
                if parent_name:
                    for part in parent_name.split('.'):
                        parent = getattr(parent, part)
                
                if isinstance(module, nn.GroupNorm):
                    new_norm = nn.InstanceNorm3d(
                        module.num_channels,
                        affine=True
                    )
                elif isinstance(module, nn.LayerNorm):
                    normalized_shape = module.normalized_shape[0] if isinstance(module.normalized_shape, tuple) else module.normalized_shape
                    new_norm = nn.InstanceNorm3d(
                        normalized_shape,
                        affine=True
                    )
                setattr(parent, attr_name, new_norm)

    def forward(self, x):
        x = self.model(x)
        return self.final_activation(x)

"""
    
class MedNextGenerator3D(nn.Module):
    def __init__(self, input_channels=2, output_channels=1):
        super(MedNextGenerator3D, self).__init__()

        self.model = create_mednext_v1(
            num_input_channels=input_channels,
            num_classes=output_channels,
            model_id='M',              
            kernel_size=3,
            deep_supervision=False,
        )

        self.final_activation = nn.Tanh()

    def forward(self, x):
        x = self.model(x)
        return self.final_activation(x)
    

"""

    
PatchDiscriminator3D = PatchDiscriminator(
    spatial_dims=3,
    num_layers_d=3,
    num_channels=96,
    in_channels=3,
    out_channels=1
)
