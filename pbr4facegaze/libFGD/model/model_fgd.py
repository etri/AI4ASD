""" 
   * Source: libFGD.model.model_fgd.py
   * License: PBR License (Dual License)
   * Modified by Howon Kim <hw_kim@etri.re.kr>
   * Date: 27 Jul 2022, ETRI
   * Copyright 2022. ETRI all rights reserved. 
"""


from __future__ import division

import torch
import torch.nn as nn
import scipy.io as sio

from libFGD.model.resnet import resnet50, BasicBlock, conv1x1



class norm3DFaceNet(nn.Module):
    
    """ norm3DFaceNet class
    Args
    """
    
    def __init__(self):

        """ init function of e2e3DFaceNet class
        
        Args:             
            _args: external args info
            
        """
        
        super(norm3DFaceNet, self).__init__()
                
        self.enc_feature_sep  = resnet50(pretrained=False)
        self.enc_camgaze_sep  = encoder_resnet_submodule_fc(BasicBlock, 1024, 512, 3, 128, 3)
        
       
    def forward(self, x):
        
        """ forward function to detect facial 3D pose 
        
        Args: 
            x: Bx3x224x224         
        """
                    
        out_E0_bx256x14x14 = self.enc_feature_sep(x)

        out_gaze_bx3 = self.enc_camgaze_sep(out_E0_bx256x14x14) 
        out_gaze_bx3 = out_gaze_bx3.detach().cpu().numpy()            

        outputs = {'pred_gaze':out_gaze_bx3}
            
        return outputs
    
        
class ortho3DFaceNet(nn.Module):
         
    """ ortho3DFaceNet class to detect facial 3D pose with orthographic camera projection model
    Args
    """
    
    def __init__(self):

        """ init function of ortho3DFaceNet class
        
        Args:             
        """
        
        super(ortho3DFaceNet, self).__init__()
        
        self.BatchNum=0        
        self.init_var()
        
        self.enc_feature  = resnet50(pretrained=True)
        self.enc_headpose = encoder_resnet_submodule_fc(BasicBlock, 1024, 512, 3, 128, 13)
        self.enc_lm3D_cam = encoder_resnet_submodule_fc(BasicBlock, 1024, 512, 3, 128, 33)
                        
        
    def init_var(self, path='./libFGD/model/param_pbr.mat'):     
        
        """ init_var function to load and set params for CNN net
        
        Args: 
            path: file path for params            
        """
        
        mat=sio.loadmat(path)
        mean_m_1x13 = mat['mean_m_ortho']
        std_m_1x13  = mat['std_m_ortho']
        mean_lm3D_can_1xp3 = mat['mean_lm3D_can']
        mean_lm3D_cam_1xp3 = mat['mean_lm3D_cam_ortho']
        std_lm3D_cam_1xp3  = mat['std_lm3D_cam_ortho']
                
        self.mean_m_1x13 = torch.from_numpy(mean_m_1x13).float()
        self.std_m_1x13  = torch.from_numpy(std_m_1x13).float()
        self.mean_lm3D_can_1xp3 = torch.from_numpy(mean_lm3D_can_1xp3).float()
        self.mean_lm3D_cam_1xp3 = torch.from_numpy(mean_lm3D_cam_1xp3).float()
        self.std_lm3D_cam_1xp3  = torch.from_numpy(std_lm3D_cam_1xp3).float()
                    
        
    def expand_var(self, batchsize):
        
        """ expand_var function to expand params according to input batch size
        
        Args: 
            BatchNum: input batch size
        """
        
        self.BatchNum = batchsize
        self.mean_m_bx13 = self.mean_m_1x13.cuda().expand(batchsize, self.mean_m_1x13.shape[1])
        self.std_m_bx13  = self.std_m_1x13.cuda().expand(batchsize, self.std_m_1x13.shape[1])
        self.mean_lm3D_can_bxp3 = self.mean_lm3D_can_1xp3.cuda().expand(batchsize, self.mean_lm3D_can_1xp3.shape[1])
        self.mean_lm3D_cam_bxp3 = self.mean_lm3D_cam_1xp3.cuda().expand(batchsize, self.mean_lm3D_cam_1xp3.shape[1])
        self.std_lm3D_cam_bxp3  = self.std_lm3D_cam_1xp3.cuda().expand(batchsize, self.std_lm3D_cam_1xp3.shape[1])
        
        
    def forward(self, x):
        
        """ forward function to detect facial 3D pose 
        
        Args: 
            x: 1x3x224x224 cropped rgb image           
        """
        
        B = x.shape[0]
        if B != self.BatchNum:       
            self.expand_var(B)
            self.BatchNum = B
            
        out_E0_bx256x14x14 = self.enc_feature(x)
        out_headpose_bx12  = self.enc_headpose(out_E0_bx256x14x14)
        out_lm3D_cam_bxp3  = self.enc_lm3D_cam(out_E0_bx256x14x14)   
                
        pred_lm3D_can_bxp3 = self.mean_lm3D_can_bxp3
        pred_lm3D_cam_bxp3 = out_lm3D_cam_bxp3 * self.std_lm3D_cam_bxp3 + self.mean_lm3D_cam_bxp3
        pred_lm3D_can_bxp3 = pred_lm3D_can_bxp3.view(self.BatchNum, -1, 3)
        pred_lm3D_cam_bxp3 = pred_lm3D_cam_bxp3.view(self.BatchNum, -1, 3)
        
        pred_m_full_bx13 = out_headpose_bx12 * self.std_m_bx13 + self.mean_m_bx13        
        pred_tvec_bx3  = pred_m_full_bx13[:, 9:12]
        pred_scale_bx1 = torch.unsqueeze(pred_m_full_bx13[:, 12], 1)
        out_rot_x = pred_m_full_bx13[:, 0:3]        
        out_rot_y = pred_m_full_bx13[:, 3:6]
        out_rot_z = pred_m_full_bx13[:, 6:9]
        out_rot_x_n = torch.div(out_rot_x, torch.norm(out_rot_x, dim=1).unsqueeze(1))
        out_rot_y_n = torch.div(out_rot_y, torch.norm(out_rot_y, dim=1).unsqueeze(1))        
        out_rot_z_n = torch.div(out_rot_z, torch.norm(out_rot_z, dim=1).unsqueeze(1))        
        pred_rmtx_bx3x3 = torch.stack([out_rot_x_n, out_rot_y_n, out_rot_z_n], dim=1)
        
        pred_cameras = [pred_rmtx_bx3x3, pred_tvec_bx3, pred_scale_bx1]    
        repred_lm3D_can_bxp3 = self.compute_transformed_lm(pred_cameras, pred_lm3D_can_bxp3)
        repred_lm3D_can_bxp3 = repred_lm3D_can_bxp3.view(self.BatchNum, -1, 3)
        
        pred_axis=self.compute_transformed_axis(pred_cameras)
        
        
        outputs = {'pred_m_full': pred_m_full_bx13,                    
                   'pred_axis': pred_axis,
                   'pred_lm3D_can': pred_lm3D_can_bxp3, 
                   'pred_lm3D_cam': pred_lm3D_cam_bxp3, 
                   'repred_lm3D_can': repred_lm3D_can_bxp3
                  }
                    
        return outputs
        
            
    def compute_transformed_lm(self, cameras, lms):
        
        """ compute_transformed_lm function to transform detected landmarks to camera coord.
        
        Args: 
            cameras: [rmtx, tvec, cam_mtx]
            lms: 1xnx3
        """
        camera_rmtx_bx3x3, camera_tvec_bx3, camera_scale_bx1 = cameras
        points_bxpx3_can = lms.view(lms.size(0), -1, 3).clone()
        
        cameratrans_rmtx_bx3x3 = camera_rmtx_bx3x3.permute(0, 2, 1)
    
        points_bxpx3_cam = torch.matmul(points_bxpx3_can.clone(), cameratrans_rmtx_bx3x3)
    
        points_bxpx3_cam = torch.einsum('bik,bj->bik', [points_bxpx3_cam, camera_scale_bx1])
        
        points_bxpx3_cam[:, :, 0:3] = points_bxpx3_cam[:, :, 0:3] + camera_tvec_bx3.view(-1, 1, 3)[:, :, 0:3]             
                
        return points_bxpx3_cam.view(points_bxpx3_cam.size(0), -1)
    
    
    def compute_transformed_axis(self, cameras):
        
        """ compute_transformed_axis function to cal. the xyz axis points at camera coord.
        
        Args: 
            cameras: [rmtx, tvec, cam_mtx]
            lms: 1xnx3
        """
        
        camera_rmtx_bx3x3, camera_tvec_bx3, camera_scale_bx1 = cameras
        cameratrans_rot_bx3x3 = camera_rmtx_bx3x3.permute(0, 2, 1)
    
        B = camera_scale_bx1.shape[0]        
        axis_len = 100.0
        points_bxpx3 = torch.tensor([[0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len]])
        points_bxpx3 = points_bxpx3.float().cuda().expand(B, 4, 3)
       
        points_bxpx3 = torch.matmul(points_bxpx3, cameratrans_rot_bx3x3)    
        xy_bxpx3 = torch.einsum('bik,bj->bik', [points_bxpx3, camera_scale_bx1])        
        xy_bxpx3[:, :, 0:3] = xy_bxpx3[:, :, 0:3] + camera_tvec_bx3.view(-1, 1, 3)[:, :, 0:3]             
       
        return xy_bxpx3[:, :, 0:2]
    
    
class encoder_resnet_submodule_fc(nn.Module):
    
    """ encoder_resnet_submodule_fc class with small stacked cnn bloacks and fc layers
    Args
    """
    
    def __init__(self, block, inplane=256, outplane=256, blocks=2, fc_mid=128, fc_out=3):
        
        """ init function of encoder_resnet_submodule_fc class
        
        Args:             
            block: func. block
            inplane: start chs of net
            outplane: end chs of net
            blocks: num of blocks to stack
            fc_mid: num of ch at fc mid layer
            fc_out: num of ch at fc out layer            
        """
        
        super(encoder_resnet_submodule_fc, self).__init__()
        
        self.layers_enc = self._make_layer(block, inplane, outplane, blocks, stride=2, dilate=False)
        
        self.layer_avgpool = nn.AdaptiveAvgPool2d((1, 1))
            
        self.layers_fc = nn.Sequential(
            nn.Linear(outplane, fc_mid),
            nn.ReLU(),            
            nn.Linear(fc_mid, fc_out)            
        )
            

    def _make_layer(self, block, inplanes, planes, blocks, stride=1, dilate=False):
        
        """ _make_layer function to stack cnn layers
        
        Args:             
            block: func. block
            inplane: start chs of net
            outplane: end chs of net
            blocks: num of blocks to stack
            stride: stride of input cnn block
            
        """
        
        norm_layer = nn.BatchNorm2d
        downsample = None
        dilation = 1
        groups=1
        base_width=64
        previous_dilation = dilation
        if dilate:
            dilation *= stride
            stride = 1
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample, groups,
                            base_width, previous_dilation, norm_layer))
        inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(inplanes, planes, groups=groups,
                                base_width=base_width, dilation=dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    
    def forward(self, x):
        
        """ forward function of cnn 
        
        Args: 
            x: 1xchxhxw            
        """
                
        out_enc = self.layers_enc(x)
        out_enc = self.layer_avgpool(out_enc)
        out_enc = out_enc.view(out_enc.size(0), -1)        
        out = self.layers_fc(out_enc)
       
        return out      
    
   
