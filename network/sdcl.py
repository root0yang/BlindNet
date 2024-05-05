from typing import Any
from packaging import version

from abc import ABC

import torch
from torch import nn
import torch.nn.functional as F

from network.mynn import initialize_weights
import numpy as np
import time


class Disentangle_Contrast(nn.Module):
    def __init__(self, args):
        super(Disentangle_Contrast, self).__init__()

        self.args = args
        self.temperature = 0.1
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')

        self.num_patch = self.args.num_patch
        self.num_classes = 19
        self.max_samples = 1000


    # reshape label or prediction
    def reshape_map(self, map, shape):

        map = map.unsqueeze(1).float().clone()
        map = torch.nn.functional.interpolate(map, shape, mode='nearest')
        map = map.squeeze(1).long()

        return map


    def _contrastive(self, pos_q, pos_k, neg):
        num_patch, _, patch_dim = pos_q.shape

        # l_pos shape : (num_patch, 1)
        l_pos = torch.bmm(pos_q, pos_k.transpose(2, 1))
        l_pos = l_pos.view(num_patch, 1)

        # l_neg shape : (num_patch, negative_size)
        l_neg = torch.bmm(pos_q, neg.transpose(2, 1))
        l_neg = l_neg.view(num_patch, -1)

        out = torch.cat([l_pos, l_neg], dim=1) / self.args.nce_T

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long, device=pos_q.device))

        return loss


    def Disentangle_Sampler(self, correct_maps, feats_q, feats_k, predicts, predicts_j, labels):
        B, HW, C = feats_q.shape
        start_ts = time.time()
        
        # X_pos_q : anchors, X_pos_k : positives, X_neg : negatives
        X_pos_q = []
        X_pos_k = []
        X_neg = []

        for ii in range(B):
            img_sample_ts = time.time()
            M = correct_maps[ii]
            # indices : wrong prediction location in query features
            indices = (M == 1).nonzero()

            classes_labels = torch.unique(labels[ii])
            classes_wrong = torch.unique(predicts_j[ii, indices])

            pos_indices = []
            neg_indices = []

            # sample anchor, positive, negative for each wrong class
            for cls_id in classes_wrong:
                sampling_time = time.time()
                # cls_indices : anchor, positive indices
                cls_indices = ((M == 1) & (predicts_j[ii] == cls_id)).nonzero()

                # pass if wrong class doesn't exist in the image
                if cls_id not in classes_labels:
                    continue
                else:
                    neg_cls_indices = (labels[ii] == cls_id).nonzero()
                
                    if neg_cls_indices.size(0) < self.num_patch:
                        continue
                    
                    neg_sampled_indices = [neg_cls_indices[torch.randperm(neg_cls_indices.size(0))[:self.num_patch]].squeeze()] * cls_indices.size(0)
                    neg_sampled_indices = torch.cat(neg_sampled_indices, dim=0)

                pos_indices.append(cls_indices)
                neg_indices.append(neg_sampled_indices)

            if not pos_indices:
                continue
            pos_indices = torch.cat(pos_indices, dim=0)
            neg_indices = torch.cat(neg_indices, dim=0)

            # anchor from query feature
            X_pos_q.append(feats_q[ii, pos_indices, :])
            # positive from key feature
            X_pos_k.append(feats_k[ii, pos_indices, :])
            # Negative from query feature
            X_neg.append(feats_q[ii, neg_indices, :].view(pos_indices.size(0), self.num_patch, C))

        if not X_pos_q:
            return None, None, None
        # X_pos_q, X_pos_k shape : (num_patch, 1, C)
        # X_neg shape : (num_patch, negative_size, C)
        X_pos_q = torch.cat(X_pos_q, dim=0)
        X_pos_k = torch.cat(X_pos_k, dim=0)
        X_neg = torch.cat(X_neg, dim=0)
        
        if X_pos_q.shape[0] > B * self.max_samples:
            indices = torch.randperm(X_pos_q.size(0))[:B*self.max_samples]
            X_pos_q = X_pos_q[indices, :, :]
            X_pos_k = X_pos_k[indices, :, :]
            X_neg = X_neg[indices, :, :]

        return X_pos_q, X_pos_k, X_neg


    def forward(self, feats_q, feats_k, predicts, predicts_j, labels):
        B, C, H, W = feats_q.shape

        # reshape the labels and predictions to feature map's size
        labels = self.reshape_map(labels, (H, W))
        predicts = self.reshape_map(predicts, (H, W))
        predicts_j = self.reshape_map(predicts_j, (H, W))
        
        # calculate Correction map
        correct_maps = torch.ones_like(predicts, device=feats_q[0].device)
        correct_maps[predicts == predicts_j] = 0
        correct_maps[labels == 255] = 0
        correct_maps[predicts != labels] = 0
        correct_maps = correct_maps.flatten(1, 2)

        predicts = predicts.flatten(1, 2)
        predicts_j = predicts_j.flatten(1, 2)
        labels = labels.flatten(1, 2)

        feats_k = feats_k.detach()

        feats_q_reshape = feats_q.permute(0, 2, 3, 1).flatten(1, 2)
        feats_k_reshape = feats_k.permute(0, 2, 3, 1).flatten(1, 2)
        
        # Sample the anchor and positives, negatives
        patches_q, patches_k, patches_neg = self.Disentangle_Sampler(correct_maps, feats_q_reshape, feats_k_reshape, 
                                                                predicts, predicts_j, labels)

        if patches_q is None:
            loss = torch.FloatTensor([0]).cuda()
            return loss

        loss = self._contrastive(patches_q, patches_k, patches_neg)

        return loss

