""" 
   * Source: libFPD.api.py
   * License: PBR License (Dual License)
   * Modified by Howon Kim <hw_kim@etri.re.kr>
   * Date: 15 Nov 2021, ETRI

"""

import json
import cv2 as cv
import numpy as np
from pathlib import Path
import math


import torch

from libFPD.model.model_fpd import ortho3DFaceNet


class Face3DPose:
    
    """ Face3DPose class
    Args:     
    """    
    
    def __init__(self, model_path):
        
        """ init function of Face3DPose class
        
        Args: 
            model_path:  file path of pretrained model            
        """
                
        self.image_scale=127.5
        self.image_bias=1.0
        self.net_input_h =224
        self.net_input_w =224
        
        # gen orthof3d model... 
        if not Path(model_path).is_file():
            print(' Incorrect ortho path....terminate process')
            return         
        checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
        
        self.model_orthof3d = ortho3DFaceNet()
        model_dict = self.model_orthof3d.state_dict()
        for k in checkpoint.keys():
            model_dict[k.replace('module.', '')] = checkpoint[k]
        
        self.load_static_dict_matchedOnly(self.model_orthof3d, model_dict)
        self.model_orthof3d = self.model_orthof3d.cuda()
        self.model_orthof3d.eval()
                
    
    def load_static_dict_matchedOnly(self, model, checkpoint):
        
        """ load_static_dict_matchedOnly function to load model's params to model
        
        Args: 
            model: CNN model
            checkpoint: saved model's params            
        """
        
        pretrained_dict = checkpoint
        model_dict = model.state_dict()

        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}        
        model_dict.update(pretrained_dict)         
        model.load_state_dict(model_dict, strict=False)   
    
    
    def predict_ortho(self, frame, roi, tgt_size=(224, 224)):
        
        """ predict_persp function to detect face 3D pose and 3D landmarks points at input image coord 
        
        Args: 
            frame: 3xhxw
            roi: [sx, sy, ex, ey]
            cam_mtx: 3x3 intrinsic matrix of input camera
            tgt_size: fixed to (224,224)
        """
        
        face=self.crop_img(frame, roi)
        scale=face.shape[0]/tgt_size[0]
        face = cv.resize(face, dsize=tgt_size, interpolation=cv.INTER_LINEAR)
        
        face = np.transpose(face, [2, 0, 1])
        face = face / self.image_scale - self.image_bias
        face_c=torch.from_numpy(face.astype(np.float32)).cuda().unsqueeze(0)
        outputs = self.model_orthof3d(face_c)
        
        m_pred =outputs['pred_m_full']
        m_pred = m_pred.squeeze().detach().cpu().numpy().astype(np.float32)
        rmtx_ortho = m_pred[0:9].reshape(3, 3)
        rmtx_ortho = self.refine_rmtx(rmtx_ortho)
        
        lm2D = outputs['pred_lm3D_cam'].squeeze().detach().cpu().numpy()[:, 0:2].astype(np.float32)
        axis = outputs['pred_axis'].squeeze().detach().cpu().numpy()[:, 0:2].astype(np.float32)
        
        lm2D *= scale
        lm2D[:, 0] += roi[0]
        lm2D[:, 1] += roi[1]
        
        axis *= scale
        axis[:, 0] += roi[0]
        axis[:, 1] += roi[1]
        
        outputs ={'lm2D': lm2D,
                  'rmtx_ortho': rmtx_ortho,
                  'axis': axis
                  }
        
        return outputs
    
    
    def predict_persp(self, frame, roi, cam_mtx, tgt_size=(224, 224)):
        
        """ predict_persp function to detect face 3D pose and 3D landmarks points at input image coord 
        
        Args: 
            frame: 3xhxw
            roi: [sx, sy, ex, ey]
            cam_mtx: 3x3 intrinsic matrix of input camera
            tgt_size: fixed to (224,224)
        """
        
        face=self.crop_img(frame, roi)
        scale=face.shape[0]/tgt_size[0]
        face = cv.resize(face, dsize=tgt_size, interpolation=cv.INTER_LINEAR)
        
        face = np.transpose(face, [2, 0, 1])
        face = face / self.image_scale - self.image_bias
        face_c=torch.from_numpy(face.astype(np.float32)).cuda().unsqueeze(0)
        outputs = self.model_orthof3d(face_c)
        
        lm3D        = outputs['pred_lm3D_can'].squeeze().detach().cpu().numpy().astype(np.float32)
        lm2D        = outputs['pred_lm3D_cam'].squeeze().detach().cpu().numpy()[:, 0:2].astype(np.float32)
        lm2D_repred = outputs['repred_lm3D_can'].squeeze().detach().cpu().numpy()[:, 0:2].astype(np.float32)
        
        lm2D *= scale
        lm2D[:, 0] += roi[0]
        lm2D[:, 1] += roi[1]
        lm2D_repred *= scale
        lm2D_repred[:, 0] += roi[0]
        lm2D_repred[:, 1] += roi[1]
        
        lm3D = lm3D
        lm2D = lm2D_repred
        
        success, rvec, tvec, inliers = cv.solvePnPRansac(lm3D, lm2D, cam_mtx, None, flags=cv.SOLVEPNP_EPNP)
        if not success:
            return None
        
        success, rvec, tvec = cv.solvePnP(lm3D, lm2D, cam_mtx, None, 
                                          rvec=rvec, tvec=tvec, 
                                          useExtrinsicGuess=True, flags=cv.SOLVEPNP_ITERATIVE)
        if not success:
            return None
        
        rmtx, _ = cv.Rodrigues(rvec)
        
        outputs ={'rmtx': rmtx,
                  'tvec': tvec, 
                  'lm3D': lm3D,
                  'lm2D': lm2D,                  
                  }
        
        return outputs
    
    
    def crop_img(self, img, roi_box):
        
        """ crop_img function
        
        Args: 
            img: 3xhxw
            roi_box: [sx, sy, ex, ey]
        """
        
        h, w = img.shape[:2]    
        sx, sy, ex, ey = [int(round(_)) for _ in roi_box[0:4]]
        dh, dw = ey - sy, ex - sx
        if len(img.shape) == 3:
            res = np.zeros((dh, dw, 3), dtype=np.uint8)
        else:
            res = np.zeros((dh, dw), dtype=np.uint8)
        if sx < 0:
            sx, dsx = 0, -sx
        else:
            dsx = 0
    
        if ex > w:
            ex, dex = w, dw - (ex - w)
        else:
            dex = dw
    
        if sy < 0:
            sy, dsy = 0, -sy
        else:
            dsy = 0
    
        if ey > h:
            ey, dey = h, dh - (ey - h)
        else:
            dey = dh
    
        res[dsy:dey, dsx:dex] = img[sy:ey, sx:ex]
        return res
     
    
    def refine_rmtx(self, rmtx):
        
        """ refine_rmtx function to refine the detected rmtx
        
        Args:             
            rmtx: 3x3
        """
        
        U, Sig, V_T = np.linalg.svd(rmtx)
        rmtx = U @ V_T
        return rmtx
    
        
    
    
    
