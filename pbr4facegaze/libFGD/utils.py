""" 
   * Source: libFGD.utils.py
   * License: PBR License (Dual License)
   * Modified by Howon Kim <hw_kim@etri.re.kr>
   * Date: 27 Jul 2022, ETRI
   * Copyright 2022. ETRI all rights reserved. 
"""

import numpy as np
import cv2 as cv


def get_projected_axis(cam_mtx, rmtx, tvec, axis_length=100):
        
    """ get_projected_axis function to get the projected each xyz axis point at image coord.
    
    Args: 
        cam_mtx: 3x3
        rmtx: 3x3
        tvec: 3x1            
    """
    
    axis = np.array([[0, 0, 0], [axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]], dtype=np.float32)
    axis_3d_cam  = np.matmul(rmtx, axis.T).T
    axis_3d_cam += tvec.T
    axis_2d_cam  = np.matmul(cam_mtx, axis_3d_cam.T)
    axis_2d_cam  = axis_2d_cam.T
    axis_2d_cam[:, 0] = axis_2d_cam[:, 0]/axis_2d_cam[:, 2]
    axis_2d_cam[:, 1] = axis_2d_cam[:, 1]/axis_2d_cam[:, 2]
    axis_2d_cam_px2  = axis_2d_cam[:, 0:2]
    
    return axis_2d_cam_px2


def get_projected_points(cam_mtx, rmtx, tvec, pt3D_px3):
    
    """ get_projected_points function to get the projected 3D points at image coord.
    
    Args: 
        cam_mtx: 3x3
        rmtx: 3x3
        tvec: 3x1           
        pt3D_px3: px3
    """
    
    pt3D  = np.matmul(rmtx, pt3D_px3.T).T
    pt3D += tvec.T
    pt2D_px3 = np.matmul(cam_mtx, pt3D.T).T
    pt2D_px3[:, 0] = pt2D_px3[:, 0]/pt2D_px3[:, 2]  
    pt2D_px3[:, 1] = pt2D_px3[:, 1]/pt2D_px3[:, 2]    
    pt2D_px2       = pt2D_px3[:, 0:2]
    
    return pt2D_px2


def plot_kpts(image, kpts, color=(0, 255, 0), radius=2):
        
    """ plot_kpts function to plot detected facial landmarks
    
    Args: 
        image: 3xhxw
        kpts: nx2            
    """

    image = image.copy()
    [h, w, c] = image.shape
    kpts = np.round(kpts).astype(np.int32)
    
    for i in range(kpts.shape[0]):
        st = kpts[i, :2]
        image = cv.circle(image, (st[0], st[1]), 1, color, radius)  
        
    return image


def plot_axis(image, kpts, mode='pred'):
    
    """ plot_axis function to plot detected facial 3D pose
    
    Args: 
        image: 3xhxw
        kpts: nx2            
    """
    
    image = image.copy()
    [h, w, c] = image.shape
    kpts = np.round(kpts).astype(np.int32)
    
    st = kpts[0, :2]
    if mode == 'label':
        ed = kpts[1, :2]
        image = cv.arrowedLine(image, (st[0], st[1]), (ed[0], ed[1]), (255, 255, 255), 2)
        ed = kpts[2, :2]
        image = cv.arrowedLine(image, (st[0], st[1]), (ed[0], ed[1]), (255, 255, 255), 2)
        ed = kpts[3, :2]
        image = cv.arrowedLine(image, (st[0], st[1]), (ed[0], ed[1]), (255, 255, 255), 2)
    else:
        ed = kpts[1, :2]
        image = cv.arrowedLine(image, (st[0], st[1]), (ed[0], ed[1]), (0, 0, 256), 2)
        ed = kpts[2, :2]
        image = cv.arrowedLine(image, (st[0], st[1]), (ed[0], ed[1]), (0, 256, 0), 2)
        ed = kpts[3, :2]
        image = cv.arrowedLine(image, (st[0], st[1]), (ed[0], ed[1]), (256, 0, 0), 2)
    
    return image


def vector_to_pitchyaw(vector):
        
    """ convert vector to pitch-yaw 
    Args:            
        vector: 3x1 vector
    Returns:
        vector: 2x1 vector
    """
    
    out = np.empty((2, 1))
    vector = np.divide(vector, np.linalg.norm(vector))
    out[0] = np.arcsin(vector[1])  # theta
    out[1] = np.arctan2(vector[0], vector[2])  # phi
    
    return out
    
    
def plot_gaze(image, pos, pitchyaw, thickness=2, color=(0, 0, 255)):
        
    """ plot gaze direction
    Args:            
        image_in: hxwx3 opencv loaded numpy image
        pos: (x,y) position in image
        pitchyaw: 2x1 vector            
    Returns:
        image: hxwx3 image
    """
    
    image_out = image.copy()
    (h, w) = image.shape[:2]
    length = min(400, max(100, w/8.0))        
    if len(image_out.shape) == 2 or image_out.shape[2] == 1:
        image_out = cv.cvtColor(image_out, cv.COLOR_GRAY2BGR)
    dx = -length * np.sin(pitchyaw[1]) * np.cos(pitchyaw[0])
    dy = -length * np.sin(pitchyaw[0])
    cv.arrowedLine(image_out, tuple(np.round(pos).astype(np.int32)), 
                   tuple(np.round([pos[0] + dx, pos[1] + dy]).astype(int)),
                   color, thickness, cv.LINE_AA, tipLength=0.2)

    return image_out
    