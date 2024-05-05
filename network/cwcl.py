# -*- coding: utf-8 -*-
from typing import Any
from packaging import version

from abc import ABC

import torch
from torch import nn
import torch.nn.functional as F


class Class_PixelNCELoss(nn.Module):
    def __init__(self, args):
        super(Class_PixelNCELoss, self).__init__()

        self.args = args
        self.ignore_label = 255

        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

        self.max_classes = self.args.contrast_max_classes
        self.max_views = self.args.contrast_max_views
    
    # reshape label or prediction
    def resize_label(self, labels, HW):
        labels = labels.unsqueeze(1).float().clone()
        labels = torch.nn.functional.interpolate(labels,
                                                HW, mode='nearest')
        labels = labels.squeeze(1).long()

        return labels

    def _hard_anchor_sampling(self, X_q, X_k, y_hat, y):
        # X : Feature map, shape:(B, h*w, C), y_hat : label, shape:(B, h*w), y : prediction, shape:(B, H*W?)
        batch_size, feat_dim = X_q.shape[0], X_q.shape[-1]

        classes = []
        num_classes = []
        total_classes = 0
        # 한 배치 내의 이미지들에 대한 label들로부터 존재하는 class들 골라내기
        for ii in range(batch_size):
            this_y = y_hat[ii]
            this_classes = torch.unique(this_y) # 텐서에서 중복된 요소 제거하여 존재하는 고유요소들 반환
            this_classes = [x for x in this_classes if x != self.ignore_label] # ignore label 제거
            this_classes = [x for x in this_classes if (this_y == x).nonzero().shape[0] > self.max_views] # class가 일정 개수 이상인 경우만 골라내기
            
            classes.append(this_classes)
            
            total_classes += len(this_classes)
            num_classes.append(len(this_classes))

        # return none if there is no class in the image
        if total_classes == 0:
            return None, None, None

        n_view = self.max_views

        # output tensors
        X_q_ = torch.zeros((batch_size, self.max_classes, n_view, feat_dim), dtype=torch.float).cuda()
        X_k_ = torch.zeros((batch_size, self.max_classes, n_view, feat_dim), dtype=torch.float).cuda()
        
        for ii in range(batch_size):
            this_y_hat = y_hat[ii]
            this_y = y[ii]
            this_classes = classes[ii]
            this_indices = []

            # if there is no class in the image, randomly sample patcthes
            if len(this_classes) == 0:
                indices = torch.arange(X_q.shape[1], device=X_q.device)
                perm = torch.randperm(X_q.shape[1], device=X_q.device)
                indices = indices[perm[:n_view * self.max_classes]]
                indices = indices.view(self.max_classes, -1)
                
                X_q_[ii, :, :, :] = X_q[ii, indices, :]
                X_k_[ii, :, :, :] = X_k[ii, indices, :]

                continue
            
            # referecne : https://github.com/tfzhou/ContrastiveSeg/tree/main
            for n, cls_id in enumerate(this_classes):
            
                if n == self.max_classes:
                    break
                
                # sample hard pathces(wrong prediction) and easy pathces(correct prediction)
                hard_indices = ((this_y_hat == cls_id) & (this_y != cls_id)).nonzero()
                easy_indices = ((this_y_hat == cls_id) & (this_y == cls_id)).nonzero()

                num_hard = hard_indices.shape[0]
                num_easy = easy_indices.shape[0]

                if num_hard >= n_view / 2 and num_easy >= n_view / 2:
                    num_hard_keep = n_view // 2
                    num_easy_keep = n_view - num_hard_keep
                elif num_hard >= n_view / 2:
                    num_easy_keep = num_easy
                    num_hard_keep = n_view - num_easy_keep
                elif num_easy >= n_view / 2:
                    num_hard_keep = num_hard
                    num_easy_keep = n_view - num_hard_keep
                
                perm = torch.randperm(num_hard)
                hard_indices = hard_indices[perm[:num_hard_keep]]
                perm = torch.randperm(num_easy)
                easy_indices = easy_indices[perm[:num_easy_keep]]
                indices = torch.cat((hard_indices, easy_indices), dim=0)
 
                X_q_[ii, n, :, :] = X_q[ii, indices, :].squeeze(1)
                X_k_[ii, n, :, :] = X_k[ii, indices, :].squeeze(1)

                this_indices.append(indices)   

            # fill the spare space with random pathces
            if len(this_classes) < self.max_classes:
                this_indices = torch.stack(this_indices)
                this_indices = this_indices.flatten(0, 1)
                
                num_remain = self.max_classes - len(this_classes)
                all_indices = torch.arange(X_q.shape[1], device=X_q[0].device)
                left_indices = torch.zeros(X_q .shape[1], device=X_q[0].device, dtype=torch.uint8)
                left_indices[this_indices] = 1
                left_indices = all_indices[~left_indices]
           
                perm = torch.randperm(len(left_indices), device=X_q[0].device)

                indices = left_indices[perm[:n_view * num_remain]]
                indices = indices.view(num_remain, -1)

                X_q_[ii, n + 1:, :, :] = X_q[ii, indices, :]
                X_k_[ii, n + 1:, :, :] = X_k[ii, indices, :]

        return X_q_, X_k_, num_classes
    
    def _contrastive(self, feats_q_, feats_k_):
        # feats shape : (B, nc, N, C)
        batch_size, num_classes, n_view, patch_dim = feats_q_.shape
        num_patches = batch_size * num_classes * n_view

        # feats shape : (B*nc*N, 1, C)
        feats_q_ = feats_q_.contiguous().view(num_patches, -1, patch_dim)
        feats_k_ = feats_k_.contiguous().view(num_patches, -1, patch_dim)

        # logit_positive : same positive patches between key and query
        # shape : (B * nc * N , 1)
        l_pos = torch.bmm(
            feats_q_, feats_k_.transpose(2, 1)
        )
        l_pos =l_pos.view(num_patches, 1)

        # feats shape : (B, nc*N, C)
        feats_q_ = feats_q_.contiguous().view(batch_size, -1, patch_dim)
        feats_k_ = feats_k_.contiguous().view(batch_size, -1, patch_dim)
        n_patches = feats_q_.shape[1]

        # logit negative shape : (B, nc*N, nc*N)
        l_neg_curbatch = torch.bmm(feats_q_, feats_k_.transpose(2, 1))
        
        # exclude same class patches 
        diag_block= torch.zeros((batch_size, n_patches, n_patches), device=feats_q_.device, dtype=torch.uint8)
        for i in range(num_classes):
            diag_block[:, i*n_view:(i+1)*n_view, i*n_view:(i+1)*n_view] = 1
        
        l_neg_curbatch = l_neg_curbatch[~diag_block].view(batch_size, n_patches, -1)

        # logit negative shape : (B*nc*N, nc*(N-1))
        l_neg = l_neg_curbatch.view(num_patches, -1)
        
        out = torch.cat([l_pos, l_neg], dim=1) / self.args.nce_T
    
        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=feats_q_.device))

        return loss

    def forward(self, feats_q, feats_k, labels=None, predict=None):
        B, C, H, W = feats_q.shape

        # resize label and prediction
        labels = self.resize_label(labels, (H, W))
        predict = self.resize_label(predict, (H, W))

        labels = labels.contiguous().view(B, -1)
        predict = predict.contiguous().view(B, -1)

        # change axis
        feats_q = feats_q.permute(0, 2, 3, 1)
        feats_q = feats_q.contiguous().view(feats_q.shape[0], -1, feats_q.shape[-1])
        
        feats_k = feats_k.detach()

        feats_k = feats_k.permute(0, 2, 3, 1)
        feats_k = feats_k.contiguous().view(feats_k.shape[0], -1, feats_k.shape[-1])
        
        # sample patches
        feats_q_, feats_k_, num_classes = self._hard_anchor_sampling(feats_q, feats_k, labels, predict)

        if feats_q_ is None:
            loss = torch.FloatTensor([0]).cuda()
            return loss

        loss = self._contrastive(feats_q_, feats_k_)

        del labels

        return loss

class Normalize(nn.Module):
    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out


class ProjectionHead(nn.Module):
    
    def __init__(self, dim_in, proj_dim=256):
        super(ProjectionHead, self).__init__()

        self.proj = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size=1),
            nn.ReLU(),
            #nn.BatchNorm2d(dim_in),
            nn.Conv2d(dim_in, proj_dim, kernel_size=1)
        )
        self.l2norm = Normalize(2)

    def forward(self, x):
        return self.l2norm(self.proj(x))
        #return F.normalize(self.proj(x), p=2, dim=1)

