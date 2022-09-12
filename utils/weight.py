import torch.nn as nn


def init_weights(module: nn.Module):
    """https://adityassrana.github.io/blog/theory/2020/08/26/Weight-Init.html

    Args:
        module (nn.Module): 
    """
    for m in module.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
