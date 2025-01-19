import torch
import torch.nn as nn
import torchvision.models.detection.backbone_utils as backbone_utils
import torchvision.models._utils as _utils
import torch.nn.functional as F
from collections import OrderedDict

from models.net import MobileNetV1 as MobileNetV1
from models.net import FPN as FPN
from models.net import SSH as SSH


class ClassHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ClassHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class ExpressionHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(ExpressionHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class IlluminationHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(IlluminationHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class PoseHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(PoseHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*2,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 2)

class BlurHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BlurHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*3,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 3)

class OcclusionHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(OcclusionHead,self).__init__()
        self.num_anchors = num_anchors
        self.conv1x1 = nn.Conv2d(inchannels,self.num_anchors*3,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()
        
        return out.view(out.shape[0], -1, 3)

class BboxHead(nn.Module):
    def __init__(self,inchannels=512,num_anchors=3):
        super(BboxHead,self).__init__()
        self.conv1x1 = nn.Conv2d(inchannels,num_anchors*4,kernel_size=(1,1),stride=1,padding=0)

    def forward(self,x):
        out = self.conv1x1(x)
        out = out.permute(0,2,3,1).contiguous()

        return out.view(out.shape[0], -1, 4)

class RetinaFace(nn.Module):
    def __init__(self, cfg = None, phase = 'train'):
        """
        :param cfg:  Network related settings.
        :param phase: train or test.
        """
        super(RetinaFace,self).__init__()
        self.phase = phase
        backbone = None
        if cfg['name'] == 'mobilenet0.25':
            backbone = MobileNetV1()
            if cfg['pretrain']:
                checkpoint = torch.load("weights/mobilenetV1X0.25_pretrain.tar", map_location=torch.device('cpu'))
                from collections import OrderedDict
                new_state_dict = OrderedDict()
                for k, v in checkpoint['state_dict'].items():
                    name = k[7:]  # remove module.
                    new_state_dict[name] = v
                # load params
                backbone.load_state_dict(new_state_dict)
        elif cfg['name'] == 'Resnet50':
            import torchvision.models as models
            backbone = models.resnet50(pretrained=cfg['pretrain'])

        self.body = _utils.IntermediateLayerGetter(backbone, cfg['return_layers'])
        in_channels_stage2 = cfg['in_channel']
        in_channels_list = [
            in_channels_stage2 * 2,
            in_channels_stage2 * 4,
            in_channels_stage2 * 8,
        ]
        out_channels = cfg['out_channel']
        self.fpn = FPN(in_channels_list,out_channels)
        self.ssh1 = SSH(out_channels, out_channels)
        self.ssh2 = SSH(out_channels, out_channels)
        self.ssh3 = SSH(out_channels, out_channels)

        self.ClassHead = self._make_class_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.BboxHead = self._make_bbox_head(fpn_num=3, inchannels=cfg['out_channel'])
        self.IlluminationHead = self._make_illumination_head(fpn_num = 3,inchannels = cfg['out_channel'])
        self.PoseHead = self._make_pose_head(fpn_num = 3,inchannels = cfg['out_channel'])
        self.BlurHead = self._make_blur_head(fpn_num = 3,inchannels = cfg['out_channel'])
        self.OcclusionHead = self._make_occlusion_head(fpn_num = 3,inchannels = cfg['out_channel'])
        self.ExpressionHead = self._make_expression_head(fpn_num = 3,inchannels = cfg['out_channel'])

    def _make_class_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        classhead = nn.ModuleList()
        for i in range(fpn_num):
            classhead.append(ClassHead(inchannels,anchor_num))
        return classhead
    
    def _make_bbox_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        bboxhead = nn.ModuleList()
        for i in range(fpn_num):
            bboxhead.append(BboxHead(inchannels,anchor_num))
        return bboxhead

    def _make_pose_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        posehead = nn.ModuleList()
        for i in range(fpn_num):
            posehead.append(PoseHead(inchannels,anchor_num))
        return posehead
    
    def _make_blur_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        blurhead = nn.ModuleList()
        for i in range(fpn_num):
            blurhead.append(BlurHead(inchannels,anchor_num))
        return blurhead

    def _make_illumination_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        illuminationhead = nn.ModuleList()
        for i in range(fpn_num):
            illuminationhead.append(IlluminationHead(inchannels,anchor_num))
        return illuminationhead
    
    def _make_occlusion_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        occlusionhead = nn.ModuleList()
        for i in range(fpn_num):
            occlusionhead.append(OcclusionHead(inchannels,anchor_num))
        return occlusionhead

    def _make_expression_head(self,fpn_num=3,inchannels=64,anchor_num=2):
        expressionhead = nn.ModuleList()
        for i in range(fpn_num):
            expressionhead.append(ExpressionHead(inchannels,anchor_num))
        return expressionhead

    def forward(self,inputs):
        out = self.body(inputs)

        # FPN
        fpn = self.fpn(out)

        # SSH
        feature1 = self.ssh1(fpn[0])
        feature2 = self.ssh2(fpn[1])
        feature3 = self.ssh3(fpn[2])
        features = [feature1, feature2, feature3]

        bbox_regressions = torch.cat([self.BboxHead[i](feature) for i, feature in enumerate(features)], dim=1)
        classifications = torch.cat([self.ClassHead[i](feature) for i, feature in enumerate(features)],dim=1)
        pose_classifications = torch.cat([self.PoseHead[i](feature) for i, feature in enumerate(features)],dim=1)
        blur_classifications = torch.cat([self.BlurHead[i](feature) for i, feature in enumerate(features)],dim=1)
        occlusion_classifications = torch.cat([self.OcclusionHead[i](feature) for i, feature in enumerate(features)],dim=1)
        expression_classifications = torch.cat([self.ExpressionHead[i](feature) for i, feature in enumerate(features)],dim=1)
        illumination_classifications = torch.cat([self.IlluminationHead[i](feature) for i, feature in enumerate(features)],dim=1)
        #ldm_regressions = torch.cat([self.LandmarkHead[i](feature) for i, feature in enumerate(features)], dim=1)

        if self.phase == 'train':
            output = (bbox_regressions, classifications, blur_classifications,expression_classifications, 
                     illumination_classifications,occlusion_classifications,pose_classifications,)
        else:
            output = (bbox_regressions, F.softmax(classifications, dim=-1), F.softmax(blur_classifications, dim=-1), F.softmax(expression_classifications, dim=-1),
                      F.softmax(illumination_classifications, dim=-1), F.softmax(occlusion_classifications, dim=-1), F.softmax(pose_classifications, dim=-1))
        return output
