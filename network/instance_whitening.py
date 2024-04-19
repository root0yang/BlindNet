import torch
import torch.nn as nn
import torch.nn.functional as F


class InstanceWhitening(nn.Module):

    def __init__(self, dim):    
        super(InstanceWhitening, self).__init__()
        self.instance_standardization = nn.InstanceNorm2d(dim, affine=False)

    def forward(self, x):

        x = self.instance_standardization(x)
        w = x

        return x, w


def get_covariance_matrix(f_map, eye=None):
    eps = 1e-5
    B, C, H, W = f_map.shape  # i-th feature size (B X C X H X W)
    HW = H * W
    if eye is None:
        eye = torch.eye(C).cuda()
    f_map = f_map.contiguous().view(B, C, -1)  # B X C X H X W > B X C X (H X W)
    f_cor = torch.bmm(f_map, f_map.transpose(1, 2)).div(HW-1) + (eps * eye)  # C X C / HW

    return f_cor, B

# Calcualate Cross Covarianc of two feature maps
# reference : https://github.com/shachoi/RobustNet
def get_cross_covariance_matrix(f_map1, f_map2, eye=None):
    eps = 1e-5
    assert f_map1.shape == f_map2.shape
    
    B, C, H, W = f_map1.shape
    HW = H*W
    
    if eye is None:
        eye = torch.eye(C).cuda()

    # feature map shape : (B,C,H,W) -> (B,C,HW)    
    f_map1 = f_map1.contiguous().view(B, C, -1) 
    f_map2 = f_map2.contiguous().view(B, C, -1)

    # f_cor shape : (B, C, C)
    f_cor = torch.bmm(f_map1, f_map2.transpose(1, 2)).div(HW-1) + (eps * eye)
    
    return f_cor, B

def cross_whitening_loss(k_feat, q_feat):
    assert k_feat.shape == q_feat.shape

    f_cor, B = get_cross_covariance_matrix(k_feat, q_feat)
    diag_loss = torch.FloatTensor([0]).cuda()

    # get diagonal values of covariance matrix
    for cor in f_cor:
        diag = torch.diagonal(cor.squeeze(dim=0), 0)
        eye = torch.ones_like(diag).cuda()
        diag_loss = diag_loss + F.mse_loss(diag, eye)
    diag_loss = diag_loss / B

    return diag_loss
