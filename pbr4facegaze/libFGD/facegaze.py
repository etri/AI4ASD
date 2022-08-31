""" 
   * Source: libFGD.facegaze.py
   * License: PBR License (Dual License)
   * Modified by Howon Kim <hw_kim@etri.re.kr>
   * Date: 27 Jul 2022, ETRI
   * Copyright 2022. ETRI all rights reserved. 
"""

import cv2 as cv
import numpy as np
from pathlib import Path
import torch

from libFGD.model.model_fgd import ortho3DFaceNet, norm3DFaceNet


class Face3DGaze:
    
    """ Face3DGaze class
    Args:     
    """    
    
    def __init__(self, model_path_fpose, model_path_fgaze):
        
        """ init function of Face3DPose class
        
        Args: 
            model_path:  file path of pretrained model            
        """
                
        self.image_scale=127.5
        self.image_bias=1.0
        self.net_input_h =224
        self.net_input_w =224
        
        self.tgt_netInSize=(224, 224)
        self.tgt_focalNorm=1000
        self.tgt_distanceNorm=800
                
        
        # gen orthof3d model... 
        if not Path(model_path_fpose).is_file():
            print(' Incorrect ortho path....terminate process')
            return         
        checkpoint_fpose = torch.load(model_path_fpose, map_location=lambda storage, loc: storage)
        
        self.model_orthof3d = ortho3DFaceNet()
        model_dict_fpose = self.model_orthof3d.state_dict()
        for k in checkpoint_fpose.keys():
            model_dict_fpose[k.replace('module.', '')] = checkpoint_fpose[k]
        
        self.load_static_dict_matchedOnly(self.model_orthof3d, model_dict_fpose)
        self.model_orthof3d = self.model_orthof3d.cuda()
        self.model_orthof3d.eval()
        
        # gen orthof3d model... 
        if not Path(model_path_fgaze).is_file():
            print(' Incorrect norm path....terminate process')
            return         
        checkpoint_fgaze = torch.load(model_path_fgaze, map_location=lambda storage, loc: storage)
        
        self.model_normf3d = norm3DFaceNet()
        model_dict_fgaze = self.model_normf3d.state_dict()
        for k in checkpoint_fgaze.keys():
            model_dict_fgaze[k.replace('module.', '')] = checkpoint_fgaze[k]
        
        self.load_static_dict_matchedOnly(self.model_normf3d, model_dict_fgaze)
        self.model_normf3d = self.model_normf3d.cuda()
        self.model_normf3d.eval()
                
    
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
    
    
    def predict_gaze(self, frame, roi, cam_mtx, tgt_size=(224, 224)):
        
        """ predict_persp function to detect face 3D pose and 3D landmarks points at input image coord 
        
        Args: 
            frame: 3xhxw
            roi: [sx, sy, ex, ey]
            cam_mtx: 3x3 intrinsic matrix of input camera
            tgt_size: fixed to (224,224)
        """
        
        #crop image for facepose infer
        face=self.crop_img(frame, roi)
        scale=face.shape[0]/tgt_size[0]
        face = cv.resize(face, dsize=tgt_size, interpolation=cv.INTER_LINEAR)
        
        #infer facepose
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
        
        #normalize image
        outputs = self.normalize_camera(frame, lm3D, cam_mtx, rmtx, tvec)
        RT_I2N  = outputs['RT_I2N']                        
        rmtx_i2n, tvec_i2n = self.get_seperate_RT_matrix(RT_I2N) 
       
        #infer gaze
        image_norm     = outputs['image']
        image_norm_c   = np.transpose(image_norm, [2, 0, 1])
        image_norm_c   = image_norm_c / self.image_scale - self.image_bias
        image_norm_c   = torch.from_numpy(image_norm_c.astype(np.float32)).cuda().unsqueeze(0)
        outputs = self.model_normf3d(image_norm_c)
        
        camgaze_norm_pred = outputs['pred_gaze'][0]        
        camgaze_pred  = np.matmul(rmtx_i2n, camgaze_norm_pred).T    
        camgaze_pred /= np.linalg.norm(camgaze_pred)
        camgaze_pred  = camgaze_pred.reshape(3, 1)
                       
        outputs ={'rmtx': rmtx,
                  'tvec': tvec, 
                  'lm3D': lm3D,
                  'lm2D': lm2D,  
                  'gaze': camgaze_pred}
        
        return outputs
        
    
    def normalize_camera(self, imageIn, lm3D_can, cam_mtx, rmtx, tvec):
        
        """ normalize input image to target distanced and focallengthed image
        
        Args: 
            imageIn: 3xhxw input image
            lm3D_can: nx3 3D points at facial coord.
            cam_mtx: 3x3 intrinsic matrix of input camera
            rmtx: 3x3 rotation matrix of face
            tvec: 3x1 translation vector of face            
        """
        
        RT_I2E = self.get_4x4_RT_matrix(rmtx, tvec)
        
        lm3D = self.transform_3Dto3D(rmtx, tvec, lm3D_can)
        face_center=np.mean(lm3D, axis=0)
            
        cam_norm= np.array([
                  [self.tgt_focalNorm, 0.0, self.tgt_netInSize[0]*0.5],
                  [0.0, self.tgt_focalNorm, self.tgt_netInSize[1]*0.5],
                  [0.0, 0.0, 1.0],
                  ])
        
        
        distance = np.linalg.norm(face_center) 
        z_scale = self.tgt_distanceNorm / distance
        S = np.array([  
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, z_scale],
        ])
        
        hRx = rmtx[:, 0]
        forward = (face_center / distance).reshape(3)
        down = np.cross(forward, hRx)
        down /= np.linalg.norm(down)
        right = np.cross(down, forward)
        right /= np.linalg.norm(right)
        R = np.c_[right, down, forward].T  
        
        image_norm, W = self.warp_perspective_image(R, S, cam_mtx, cam_norm, self.tgt_netInSize, imageIn)
                
        RT_I2N, RT_N2E= self.get_Transformed2NormCam_RT_matrices(R, distance, self.tgt_distanceNorm, RT_I2E) 
                
        outputs={
            'image': image_norm, 
            'cam_mtx':cam_norm,
            'RT_I2N':RT_I2N,            
            }
        
        return outputs
    
    
    def get_4x4_RT_matrix(self, rmtx_I2T, tvec_I2T):
        
        """ get_4x4_RT_matrix function
        Args:                        
            rmtx_I2T: 3x3 rotation matrix
            tvec_I2T: 3x1 translation vector
        Return:
            RT_I2T: 4x4 transformation matrix
        """
        
        RT_I2T=np.zeros((4,4), dtype=np.float32)
        RT_I2T[0:3, 0:3]=rmtx_I2T
        RT_I2T[0:3, 3]=tvec_I2T.reshape(3)
        RT_I2T[3, 3]=1.0
        
        return RT_I2T
    
    
    def get_seperate_RT_matrix(self, RT_mtx):
        
        """ get_seperate_RT_matrix function
        Args:                        
            RT_mtx: 4x4 transformation matrix            
        Return:
            rmtx: 3x3 rotation matrix
            tvec: 3x1 translation vector            
        """
        
        rmtx=RT_mtx[0:3, 0:3]
        tvec=RT_mtx[0:3, 3].reshape(-1, 1)
        return rmtx, tvec
    
    
    def transform_3Dto3D(self, rmtx, tvec, pt3D_px3):
        
        """ transform_3Dto3D function
        Args:                        
            rmtx: 3x3 rotation matrix
            tvec: 3x1 translation vector            
            pt3D_px3: px3 3D points
            
        Return:
            pt3D_px3_out: px3 3D points
        """
        
        pt3D_px3_out = np.matmul(rmtx, pt3D_px3.T).T
        pt3D_px3_out += tvec.T
        return pt3D_px3_out 
    
    
    def warp_perspective_image(self, R, S, cam_mtx, cam_mtx_norm, roiSize, image):
        
        """ warp_perspective_image function
        Args:                        
            R: 3x3 rotation matrix
            S: 3x3 scaling matrix
            cam_mtx: 3x3 camera matrix
            cam_mtx_norm: tgt 3x3 norm camera matrix
            roiSize: (w,h) roi size 
            image: hxwx3 image                       
        Return:
            img_warped: hxwx3 warped image
            W: 3x3 warp matrix
        """
        
        W = np.dot(np.dot(cam_mtx_norm, S), np.dot(R, np.linalg.inv(cam_mtx)))
        img_warped = cv.warpPerspective(image, W, roiSize)
        return img_warped, W
           
    
    def get_Transformed2NormCam_RT_matrices(self, tgt_R, cur_dist, tgt_dist, RT_I2F) :
        
        """ warp_perspective_image function
        Args:                        
            tgt_R: 3x3 tgt rotation matrix
            cur_dist: current distance btw. cam and face
            tgt_dist: tgt distance
            RT_I2F: 4x4 transformation matrix btw. cam and face            
        Return:
            RT_I2N_cal: 4x4 transformation matrix btw. cam and norm cam
            RT_N2F_cal: 4x4 transformation matrix btw. norm cam and face
        """
        
        RT_I2N_cal = np.zeros((4, 4), dtype=np.float32)
        RT_I2N_cal[0:3, 0:3] = tgt_R.T
        tvec_N = np.array([0, 0, cur_dist-tgt_dist])
        tvec_I2N = tgt_R.T.dot(tvec_N)        
        RT_I2N_cal[0:3, 3] = tvec_I2N        
        RT_I2N_cal[3, 3] = 1.0        
        RT_N2F_cal = np.linalg.inv(RT_I2N_cal).dot(RT_I2F)
        
        return RT_I2N_cal, RT_N2F_cal
    
    
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
     
    
        
    
    
    
