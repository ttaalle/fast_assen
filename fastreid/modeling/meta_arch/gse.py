# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""
import torch
from torch import nn
import math
import pdb

def normalize(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x


def euclidean_dist(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def normalize_rank(x, axis=-1):
    """Normalizing to unit length along the specified dimension.
    Args:
      x: pytorch Variable
    Returns:
      x: pytorch Variable, same shape as input
    """
    x = 1. * x / (torch.norm(x, 2, axis, keepdim=True).expand_as(x) + 1e-12)
    return x

def euclidean_dist_rank(x, y):
    """
    Args:
      x: pytorch Variable, with shape [m, d]
      y: pytorch Variable, with shape [n, d]
    Returns:
      dist: pytorch Variable, with shape [m, n]
    """
    m, n = x.size(0), y.size(0)
    xx = torch.pow(x, 2).sum(1, keepdim=True).expand(m, n)
    yy = torch.pow(y, 2).sum(1, keepdim=True).expand(n, m).t()
    dist = xx + yy
    dist.addmm_(1, -2, x, y.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    return dist

def gse_loss(dist_mat, dist_mat_st, dist_mat_at, labels, margin,alpha,tval):
    """
    Args:
      dist_mat: pytorch Variable, pair wise distance between samples, shape [N, N]
      labels: pytorch LongTensor, with shape [N]
      
    """
    assert len(dist_mat.size()) == 2
    assert dist_mat.size(0) == dist_mat.size(1)
    N = dist_mat.size(0)
    #dist_mat_st = torch.from_numpy(dist_mat_st_np)
    #dist_mat_at = torch.from_numpy(dist_mat_at_np)

    total_loss = 0.0
    for ind in range(N):
        is_pos = labels.eq(labels[ind])
        is_pos[ind] = 0
        is_neg = labels.ne(labels[ind])
        
        dist_ap = dist_mat[ind][is_pos]
        dist_an = dist_mat[ind][is_neg]

        dist_ap_st = dist_mat_st[ind][is_pos]
        dist_an_at = dist_mat_at[ind][is_neg]
        
        ap_is_pos = torch.clamp(torch.add(dist_ap,margin-alpha),min=0.0)
        ap_st_weight = torch.exp(tval*(-1/(dist_ap_st+1e-5)))
        ap_pos_num = ap_is_pos.size(0) +1e-5
        ap_pos_val_sum = torch.sum(ap_is_pos*ap_st_weight)
        loss_ap = torch.div(ap_pos_val_sum,float(ap_pos_num))

        an_is_pos = torch.lt(dist_an,alpha)
        an_less_alpha = dist_an[an_is_pos]
        weight_less_alpha = dist_an_at[an_is_pos]
        an_weight = torch.exp(tval*(-(weight_less_alpha +1e-5)))
        an_pos_num = an_is_pos.size(0) +1e-5
        an_dist_lm = alpha - an_less_alpha
        an_ln_sum = torch.sum(torch.mul(an_dist_lm,an_weight))
        loss_an = torch.div(an_ln_sum,an_pos_num )

        
        total_loss = total_loss + loss_ap + loss_an
        #pdb.set_trace()
    total_loss = total_loss*1.0/N
    return total_loss

class Gse_Loss(object):
    "GSE Loss"
    
    def __init__(self, margin=0.7, alpha=1, tval=1.0):
        self.margin = margin
        self.alpha = alpha
        self.tval = tval
        
    def __call__(self, global_feat, labels, cam_feat, view_feat, type_feat, color_feat, normalize_feature=True):
        if normalize_feature:
            global_feat = normalize_rank(global_feat, axis=-1)
            cam_feat    = normalize_rank(cam_feat, axis=-1)
            view_feat   = normalize_rank(view_feat, axis=-1)
            type_feat   = normalize_rank(type_feat, axis=-1)
            color_feat  = normalize_rank(color_feat, axis=-1)

        dist_mat = euclidean_dist_rank(global_feat, global_feat)
        dist_mat_st = (euclidean_dist_rank(cam_feat, cam_feat) + euclidean_dist_rank(view_feat, view_feat))/2
        dist_mat_at = (euclidean_dist_rank(type_feat, type_feat) + euclidean_dist_rank(color_feat, color_feat))/2
        #dist_mat_st_np = dist_mat_st.cpu().data.numpy()
        #dist_mat_at_np = dist_mat_at.cpu().data.numpy()
        total_loss = gse_loss(dist_mat, dist_mat_st, dist_mat_at, labels,self.margin,self.alpha,self.tval)
        
        return total_loss
