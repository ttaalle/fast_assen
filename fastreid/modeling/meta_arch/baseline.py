# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch
from torch import nn

from fastreid.config import configurable
from fastreid.modeling.backbones import build_backbone
from fastreid.modeling.heads import build_heads
from fastreid.modeling.losses import *
from .build import META_ARCH_REGISTRY
from .gse import Gse_Loss
import pdb

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)
    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)

@META_ARCH_REGISTRY.register()
class Baseline(nn.Module):
    """
    Baseline architecture. Any models that contains the following two components:
    1. Per-image feature extraction (aka backbone)
    2. Per-image feature aggregation and loss computation
    """

    @configurable
    def __init__(
            self,
            *,
            backbone,
            heads,
            pixel_mean,
            pixel_std,
            loss_kwargs=None
    ):
        """
        NOTE: this interface is experimental.

        Args:
            backbone:
            heads:
            pixel_mean:
            pixel_std:
        """
        super().__init__()
        # backbone
        self.backbone = backbone

        # head
        self.heads = heads

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.in_planes1 = 2048
        self.in_planes2 = 2048
        self.bn1 = nn.BatchNorm2d(self.in_planes2)
        self.relu = nn.ReLU(inplace=True)
        self.conv_cam = nn.Conv2d(self.in_planes1, self.in_planes2, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.conv_view = nn.Conv2d(self.in_planes1, self.in_planes2, kernel_size=1, stride=1, padding=0,
                               bias=False)
        #self.conv_time = nn.Conv2d(self.in_planes1, self.in_planes2, kernel_size=1, stride=1, padding=0,
        #                       bias=False)
        #self.conv_model = nn.Conv2d(self.in_planes1, self.in_planes2, kernel_size=1, stride=1, padding=0,
        #                       bias=False)
        self.conv_color = nn.Conv2d(self.in_planes1, self.in_planes2, kernel_size=1, stride=1, padding=0,
                               bias=False)
        self.conv_type = nn.Conv2d(self.in_planes1, self.in_planes2, kernel_size=1, stride=1, padding=0,
                               bias=False)

        self.classifier_cam = nn.Linear(self.in_planes2, 20, bias=False) #cam (20 for veri, 174 for wild)
        self.classifier_view = nn.Linear(self.in_planes2, 5, bias=False)  #view(5 for veri, 1 for wild)
        #self.classifier_time = nn.Linear(self.in_planes2, 1, bias=False)  #time(1 for veri, 24 for wild)
        #self.classifier_model = nn.Linear(self.in_planes2, 1, bias=False) #moid (1 for veri, 153 for wild)
        self.classifier_color = nn.Linear(self.in_planes2, 10, bias=False)  #color(10 for veri,12 for wild)
        self.classifier_type = nn.Linear(self.in_planes2, 9, bias=False) #type(9 for veri, 14 for wild)

        self.classifier_cam.apply(weights_init_classifier)
        self.classifier_view.apply(weights_init_classifier)
        #self.classifier_time.apply(weights_init_classifier)
        #self.classifier_model.apply(weights_init_classifier)
        self.classifier_color.apply(weights_init_classifier)
        self.classifier_type.apply(weights_init_classifier)
        self.conv_cam.apply(weights_init_kaiming)
        self.conv_view.apply(weights_init_kaiming)
        #self.conv_time.apply(weights_init_kaiming)
        #self.conv_model.apply(weights_init_kaiming)
        self.conv_color.apply(weights_init_kaiming)
        self.conv_type.apply(weights_init_kaiming)
        #self.veri_fe.apply(weights_init_kaiming)

        self.loss_kwargs = loss_kwargs
        self.gse = Gse_Loss()

        self.register_buffer('pixel_mean', torch.Tensor(pixel_mean).view(1, -1, 1, 1), False)
        self.register_buffer('pixel_std', torch.Tensor(pixel_std).view(1, -1, 1, 1), False)

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        heads = build_heads(cfg)
        return {
            'backbone': backbone,
            'heads': heads,
            'pixel_mean': cfg.MODEL.PIXEL_MEAN,
            'pixel_std': cfg.MODEL.PIXEL_STD,
            'loss_kwargs':
                {
                    # loss name
                    'loss_names': cfg.MODEL.LOSSES.NAME,

                    # loss hyperparameters
                    'ce': {
                        'eps': cfg.MODEL.LOSSES.CE.EPSILON,
                        'alpha': cfg.MODEL.LOSSES.CE.ALPHA,
                        'scale': cfg.MODEL.LOSSES.CE.SCALE
                    },
                    'tri': {
                        'margin': cfg.MODEL.LOSSES.TRI.MARGIN,
                        'norm_feat': cfg.MODEL.LOSSES.TRI.NORM_FEAT,
                        'hard_mining': cfg.MODEL.LOSSES.TRI.HARD_MINING,
                        'scale': cfg.MODEL.LOSSES.TRI.SCALE
                    },
                    'circle': {
                        'margin': cfg.MODEL.LOSSES.CIRCLE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.CIRCLE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.CIRCLE.SCALE
                    },
                    'cosface': {
                        'margin': cfg.MODEL.LOSSES.COSFACE.MARGIN,
                        'gamma': cfg.MODEL.LOSSES.COSFACE.GAMMA,
                        'scale': cfg.MODEL.LOSSES.COSFACE.SCALE
                    }
                }
        }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs):
        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images)

        if self.training:
            assert "targets" in batched_inputs, "Person ID annotation are missing in training!"
            targets = batched_inputs["targets"]

            # PreciseBN flag, When do preciseBN on different dataset, the number of classes in new dataset
            # may be larger than that in the original dataset, so the circle/arcface will
            # throw an error. We just set all the targets to 0 to avoid this problem.
            if targets.sum() < 0: targets.zero_()

            camid = batched_inputs["camids"]
            viewid = batched_inputs["viewids"]
            #timeid = batched_inputs["timeids"]
            modid = batched_inputs["modids"]
            colorid = batched_inputs["colorids"]
            typeid = batched_inputs["typeids"]

            a = features
            a_cam =  self.conv_cam(a)
            a_cam = self.bn1(a_cam)   
            a_cam = self.relu(a_cam)  
            #a_cams = torch.sigmoid(a_cam) 
            #a_cams1 = a_cams.mean(1).unsqueeze(1)

            a_view = self.conv_view(a)
            a_view = self.bn1(a_view)   
            a_view = self.relu(a_view)
            #a_views = torch.sigmoid(a_view) 
            #a_views1 = a_views.mean(1).unsqueeze(1)  

            #a_time = self.conv_time(a1)                   
            #a_model = self.conv_model(a1)

            a_color = self.conv_color(a)
            a_color = self.bn1(a_color)   
            a_color = self.relu(a_color) 
            #a_colors = torch.sigmoid(a_color) 
            #a_colors1 = a_colors.mean(1).unsqueeze(1)

            a_type = self.conv_type(a)
            a_type = self.bn1(a_type)   
            a_type = self.relu(a_type)  
            #a_types = torch.sigmoid(a_type)
            #a_types1 = a_types.mean(1).unsqueeze(1)

            cam_feat = self.gap(a_cam)  # (b, 2048, 1, 1)
            cam_feat = cam_feat.view(cam_feat.shape[0], -1)  # flatten to (bs, 2048)      
            view_feat = self.gap(a_view)  # (b, 2048, 1, 1)
            view_feat = view_feat.view(view_feat.shape[0], -1)  # flatten to (bs, 2048)
            #time_feat = self.gap(a_time)  # (b, 2048, 1, 1)
            #time_feat = time_feat.view(time_feat.shape[0], -1)  # flatten to (bs, 2048) 
            #model_feat = self.gap(a_model)  # (b, 2048, 1, 1)
            #model_feat = model_feat.view(model_feat.shape[0], -1)  # flatten to (bs, 2048)    
            color_feat = self.gap(a_color)  # (b, 2048, 1, 1)
            color_feat = color_feat.view(color_feat.shape[0], -1)  # flatten to (bs, 2048) 
            type_feat = self.gap(a_type)  # (b, 2048, 1, 1)
            type_feat = type_feat.view(type_feat.shape[0], -1)  # flatten to (bs, 2048)

            global_feat = self.gap(a)  # (b, 2048, 1, 1)
            global_feat = global_feat.view(global_feat.shape[0], -1)  # flatten to (bs, 2048)

            at_exp = (torch.exp(type_feat.std(0)).sum() + torch.exp(color_feat.std(0)).sum())/(features.size(1))
            st_exp = (torch.exp(cam_feat.std(0)).sum() + torch.exp(view_feat.std(0)).sum())/(features.size(1))
             
            SC_score =1- at_exp/(at_exp + st_exp)
            cam_score = self.classifier_cam(cam_feat)
            view_score = self.classifier_view(view_feat)
            #time_score = self.classifier_time(time_feat)
            #model_score = self.classifier_model(model_feat) 
            color_score = self.classifier_color(color_feat)
            type_score = self.classifier_type(type_feat)
            loss1 = 0.5*torch.nn.functional.cross_entropy(cam_score, camid) + 0.5*torch.nn.functional.cross_entropy(view_score, viewid) + 0.5*torch.nn.functional.cross_entropy(color_score, colorid) + 0.5*torch.nn.functional.cross_entropy(type_score, typeid)
            loss2 = SC_score
            loss3 = 0.3*self.gse(global_feat, targets, cam_feat, view_feat, type_feat, color_feat)
            loss2 = loss2+loss3

            outputs = self.heads(features, targets)
            losses = self.losses(outputs, targets)
            return losses, loss1, loss2
        else:
            outputs = self.heads(features)
            return outputs

    def preprocess_image(self, batched_inputs):
        """
        Normalize and batch the input images.
        """
        if isinstance(batched_inputs, dict):
            images = batched_inputs['images']
        elif isinstance(batched_inputs, torch.Tensor):
            images = batched_inputs
        else:
            raise TypeError("batched_inputs must be dict or torch.Tensor, but get {}".format(type(batched_inputs)))

        images.sub_(self.pixel_mean).div_(self.pixel_std)
        return images

    def losses(self, outputs, gt_labels):
        """
        Compute loss from modeling's outputs, the loss function input arguments
        must be the same as the outputs of the model forwarding.
        """
        # model predictions
        # fmt: off
        pred_class_logits = outputs['pred_class_logits'].detach()
        cls_outputs       = outputs['cls_outputs']
        pred_features     = outputs['features']
        # fmt: on

        # Log prediction accuracy
        log_accuracy(pred_class_logits, gt_labels)

        loss_dict = {}
        loss_names = self.loss_kwargs['loss_names']

        if 'CrossEntropyLoss' in loss_names:
            ce_kwargs = self.loss_kwargs.get('ce')
            loss_dict['loss_cls'] = cross_entropy_loss(
                cls_outputs,
                gt_labels,
                ce_kwargs.get('eps'),
                ce_kwargs.get('alpha')
            ) * ce_kwargs.get('scale')

        if 'TripletLoss' in loss_names:
            tri_kwargs = self.loss_kwargs.get('tri')
            loss_dict['loss_triplet'] = triplet_loss(
                pred_features,
                gt_labels,
                tri_kwargs.get('margin'),
                tri_kwargs.get('norm_feat'),
                tri_kwargs.get('hard_mining')
            ) * tri_kwargs.get('scale')

        if 'CircleLoss' in loss_names:
            circle_kwargs = self.loss_kwargs.get('circle')
            loss_dict['loss_circle'] = pairwise_circleloss(
                pred_features,
                gt_labels,
                circle_kwargs.get('margin'),
                circle_kwargs.get('gamma')
            ) * circle_kwargs.get('scale')

        if 'Cosface' in loss_names:
            cosface_kwargs = self.loss_kwargs.get('cosface')
            loss_dict['loss_cosface'] = pairwise_cosface(
                pred_features,
                gt_labels,
                cosface_kwargs.get('margin'),
                cosface_kwargs.get('gamma'),
            ) * cosface_kwargs.get('scale')

        return loss_dict

def rank_loss(dist_mat, dist_mat_st, dist_mat_at, labels, margin,alpha,tval):
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

class RankedLoss(object):
    "Ranked_List_Loss_for_Deep_Metric_Learning_CVPR_2019_paper"
    
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
        total_loss = rank_loss(dist_mat, dist_mat_st, dist_mat_at, labels,self.margin,self.alpha,self.tval)
        
        return total_loss
