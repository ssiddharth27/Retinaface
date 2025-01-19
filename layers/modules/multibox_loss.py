import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp
from data import cfg_mnet
GPU = cfg_mnet['gpu_train']

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size,num_priors,num_classes)
                loc shape: torch.size(batch_size,num_priors,4)
                priors shape: torch.size(num_priors,4)
            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size,num_objs,5] (last idx is the label).
        """

        loc_data, conf_data, blur_data, expression_data, illumination_data, occlusion_data, pose_data = predictions
        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))

       
        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        exp_t = torch.Tensor(num, num_priors)
        pos_t = torch.Tensor(num,num_priors)
        illu_t = torch.Tensor(num,num_priors)
        conf_t = torch.LongTensor(num, num_priors)
        blu_t = torch.LongTensor(num, num_priors)
        occl_t = torch.LongTensor(num, num_priors)

        for idx in range(num):
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, -1].data
            blur = targets[idx][:,4].data
            occlusion = targets[idx][:,7].data
            expression = targets[idx][:,5].data
            pose = targets[idx][:,8].data
            illumination = targets[idx][:,6].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels,blur,expression,illumination,occlusion,pose ,loc_t, conf_t,blu_t,exp_t,illu_t,occl_t,pos_t, idx)

        if GPU:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
        

        zeros = torch.tensor(0).cuda()
       
        pos = conf_t != zeros
        conf_t[pos] = 1

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

       
        #blur loss
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(blur_data)
        blu_p = blur_data[pos_idx]
        blu_t = blu_t[pos]
        blu_p = blu_p.view(len(blu_t),3)
        blu_t = blu_t.type(torch.LongTensor)
        blu_t = blu_t.cuda()
        loss_b = F.cross_entropy(blu_p, blu_t, reduction='sum')

        #expresion loss
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(expression_data)
        exp_p = expression_data[pos_idx]
        exp_t = exp_t[pos]
        exp_p = exp_p.view(len(exp_t),2)
        exp_t = exp_t.type(torch.LongTensor)
        exp_t = exp_t.cuda()
        loss_e = F.cross_entropy(exp_p, exp_t, reduction='sum')

        #occlution loss
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(occlusion_data)
        occl_p = occlusion_data[pos_idx]
        occl_t = occl_t[pos]
        occl_p = occl_p.view(len(occl_t),3)
        occl_t = occl_t.type(torch.LongTensor)
        occl_t = occl_t.cuda()
        loss_o = F.cross_entropy(occl_p, occl_t, reduction='sum')
        
        #pose loss
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(pose_data)
        pos_p = pose_data[pos_idx]
        pos_t = pos_t[pos]
        pos_p = pos_p.view(len(pos_t),2)
        pos_t = pos_t.type(torch.LongTensor)
        pos_t = pos_t.cuda()
        loss_p = F.cross_entropy(pos_p, pos_t, reduction='sum')

        # illumination loss
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(illumination_data)
        illu_p = illumination_data[pos_idx]
        illu_t = illu_t[pos]
        illu_p = illu_p.view(len(illu_t),2)
        illu_t = illu_t.type(torch.LongTensor)
        illu_t = illu_t.cuda()
        loss_i = F.cross_entropy(illu_p, illu_t, reduction='sum')

        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        loss_e /= N
        loss_p /= N
        loss_i /= N
        loss_o /= N
        loss_b /= N

        return loss_l, loss_c,loss_b,loss_e,loss_i,loss_o,loss_p