# -*- coding: utf-8 -*
import torch.nn as nn
import torch
import torch.nn.functional as F
import warnings

warnings.filterwarnings("error")


def cos_similar(p, q):
    sim_matrix = p.matmul(q.transpose(-2, -1))
    a = torch.norm(p, p=2, dim=-1)
    b = torch.norm(q, p=2, dim=-1)
    sim_matrix /= a.unsqueeze(-1)
    sim_matrix /= b.unsqueeze(-2)
    return sim_matrix


class ECCLoss(nn.Module):
    def __init__(self, num_class, dim):
        super().__init__()
        self.feature_table = torch.rand((num_class, dim), requires_grad=False).cuda()
        self.count = torch.zeros((num_class, 1), requires_grad=False).cuda()
        self.logit_table = torch.rand((num_class, num_class), requires_grad=False).cuda()
        self.num_class = num_class
        self.dim = dim

    def forward(self, feature, logits, targets):
        feature_copy = feature.clone().detach()
        logit_copy = logits.clone().detach()
        for i, index in enumerate(targets, start=0):
            self.feature_table[index] = self.feature_table[index] * self.count[index].expand(self.dim) + feature_copy[i]
            self.logit_table[index] = self.logit_table[index] * self.count[index].expand(self.num_class) + logit_copy[i]

            self.count[index] = self.count[index] + 1
            self.feature_table[index] = self.feature_table[index] / self.count[index]
            self.logit_table[index] = self.logit_table[index] / self.count[index]

        target_feature = self.feature_table[targets]
        feature_center_loss = (1 - torch.cosine_similarity(target_feature, feature, dim=1)).sum()

        class_table = cos_similar(self.feature_table, self.feature_table)
        class_table = (class_table - torch.min(class_table)) / (torch.max(class_table) - torch.min(class_table))
        class_table = class_table - torch.diag_embed(torch.diag(class_table))
        similar_class_value, similar_class = torch.max(class_table, dim=1)
        similar_class_feature = self.feature_table[similar_class]
        similar_target_value = similar_class_value[targets].view(-1, 1)
        feature_intra_loss = (torch.cosine_similarity(feature_copy, similar_class_feature[targets], dim=1) * similar_target_value).sum()

        target_logit = self.logit_table[targets]
        logit_center_loss = F.kl_div(F.log_softmax(logits, dim=1), F.softmax(target_logit, dim=1), reduction='sum')

        return feature_center_loss+feature_intra_loss, logit_center_loss, self.feature_table, self.logit_table