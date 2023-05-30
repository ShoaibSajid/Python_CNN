from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from loss import build_target, yolo_loss



class Yolov2_lastlayer(nn.Module):

    num_classes = 20
    num_anchors = 5

    def __init__(self, classes=None, weights_file=False):
        super(Yolov2_lastlayer, self).__init__()
        self.conv9 = nn.Sequential(nn.Conv2d(1024, (5 + self.num_classes) * self.num_anchors, kernel_size=1))
        
        self.layer_names = ['conv_bn_relu_9', 'output']
        
        self.parameters_dir = "./parameter_torch"
        Path(self.parameters_dir).mkdir(parents=True, exist_ok=True)

    def forward(self, x, gt_boxes=None, gt_classes=None, num_boxes=None, training=False):
        """
        x: Variable
        gt_boxes, gt_classes, num_boxes: Tensor
        """
          
        

        out = self.conv9(x)
        with open(self.parameters_dir+f'/layer9(conv_bn_relu)', mode='w') as f: 
            f.write(str(out))


        # out -- tensor of shape (B, num_anchors * (5 + num_classes), H, W)
        bsize, _, h, w = out.size()

        # 5 + num_class tensor represents (t_x, t_y, t_h, t_w, t_c) and (class1_score, class2_score, ...)
        # reorganize the output tensor to shape (B, H * W * num_anchors, 5 + num_classes)
        out = out.permute(0, 2, 3, 1).contiguous().view(bsize, h * w * self.num_anchors, 5 + self.num_classes)

        # activate the output tensor
        # `sigmoid` for t_x, t_y, t_c; `exp` for t_h, t_w;
        # `softmax` for (class1_score, class2_score, ...)

        xy_pred     = torch.sigmoid(out[:, :, 0:2])
        hw_pred     = torch.exp(out[:, :, 2:4])
        conf_pred   = torch.sigmoid(out[:, :, 4:5])
        class_score = out[:, :, 5:]                           # Confidence Loss
        class_pred  = F.softmax(class_score, dim=-1)          # Classification Loss
        delta_pred  = torch.cat([xy_pred, hw_pred], dim=-1)   # IOU Loss

        # if training:
        #     output_variable = (delta_pred, conf_pred, class_score)
        #     output_data     = [v.data for v in output_variable]
        #     gt_data         = (gt_boxes, gt_classes, num_boxes)
        #     target_data     = build_target(output_data, gt_data, h, w)

        #     target_variable = [v for v in target_data]
        #     box_loss, iou_loss, class_loss = yolo_loss(output_variable, target_variable)

        #     return box_loss, iou_loss, class_loss

        return delta_pred, conf_pred, class_pred
