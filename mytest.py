import os
import torch
from torchvision import transforms
from torchvision.utils import save_image
from typing import Tuple
from PIL import Image
import os
import numpy as np
from kitti_utils import generate_depth_map

from matplotlib import pyplot as plt


def read_calib_file(path):
    float_chars = set("0123456789.e+- ")
    data = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':', 1)
            value = value.strip()
            data[key] = value.split()
            if float_chars.issuperset(value):
                # try to cast to float array
                try:
                    data[key] = np.array([float(num) for num in value.split(' ')])
                except ValueError:
                    pass  # casting error: data[key] already eq. value, so pass
    return data

def getCalibInfo(calibDir):
        #retrive calibration data
        cam2cam = read_calib_file(os.path.join(calibDir, "calib_cam_to_cam.txt"))
        P_rectL = cam2cam['P_rect_02'].reshape(3, 4)
        P_rectR = cam2cam['P_rect_03'].reshape(3, 4)
        L_Kmat = cam2cam['K_02'].reshape(3,3)
        #R_Kmat = cam2cam['K_03'].reshape(3,3)
        focalLength : torch.Tensor = torch.Tensor([L_Kmat[0, 0]/1242])

        # Compute the rectified extrinsics from cam0 to camN
        T2 = np.eye(4)
        T2[0, 3] = P_rectL[0, 3] / P_rectL[0, 0]
        T3 = np.eye(4)
        T3[0, 3] = P_rectR[0, 3] / P_rectR[0, 0]

        # Compute the velodyne to rectified camera coordinate transforms
        velo2cam = read_calib_file(os.path.join(calibDir, 'calib_velo_to_cam.txt'))
        velo2cam = np.hstack((velo2cam['R'].reshape(3, 3), velo2cam['T'][..., np.newaxis]))
        T_cam0_velo = np.vstack((velo2cam, np.array([0, 0, 0, 1.0])))
        T_cam2_velo = T2.dot(T_cam0_velo)
        T_cam3_velo = T3.dot(T_cam0_velo)
        p_cam = np.array([0, 0, 0, 1])
        p_velo2 = np.linalg.inv(T_cam2_velo).dot(p_cam)
        p_velo3 = np.linalg.inv(T_cam3_velo).dot(p_cam)
        
        baseline : torch.Tensor = torch.Tensor([np.linalg.norm(p_velo3 - p_velo2)])   # rgb baseline

        return focalLength, baseline

"""
cam 2 = left cam color
cam 3 = right cam color
"""
(L_imgPath, R_imgPath, veloPath, calibDir) = ('kitti_data\\2011_09_26\\2011_09_26_drive_0036_sync\\image_02\\data\\0000000000.png', 'kitti_data\\2011_09_26\\2011_09_26_drive_0036_sync\\image_03\\data\\0000000000.png', 'kitti_data\\2011_09_26\\2011_09_26_drive_0036_sync\\velodyne_points\\data\\0000000000.bin', 'kitti_data\\2011_09_26')
#Get focalLength and baseline
focalLength, baseline = getCalibInfo(calibDir)
#get images
imgL : Image = Image.open(L_imgPath)
imgR : Image = Image.open(R_imgPath)
#conversion
convert_tensor = transforms.ToTensor()
imgL : torch.Tensor = convert_tensor(imgL).float()     #tensor
imgR : torch.Tensor = convert_tensor(imgR).float()    #tensor
#check if need to resize
resizeT = transforms.Resize((375, 1242))
#print(f"on index: {index} has shape: {imgL.size()} and {imgR.size()}")
if imgL.size() != (3, 375, 1242):
    imgL = resizeT(imgL)
    #print("new LSize:", imgL.size())
if imgR.size() != (3, 375, 1242):
    imgR = resizeT(imgR)
    #print("new RSize:", imgR.size())
#retrieve depth data
depth_gtL = generate_depth_map(calibDir, velo_filename=veloPath, cam = 2)
depth_gtR = generate_depth_map(calibDir, velo_filename=veloPath, cam = 3)


print(np.max(depth_gtL))
print(np.min(depth_gtL))
print(np.count_nonzero(depth_gtL))


#convert to tensor
depth_gtL : torch.Tensor = torch.Tensor(depth_gtL)
depth_gtR : torch.Tensor = torch.Tensor(depth_gtR)
#print(f"has shape: {depth_gtL.size()} and {depth_gtR.size()}")
if depth_gtL.size() != (375, 1242):
    depth_gtL = resizeT(depth_gtL.unsqueeze(0))
    depth_gtL = depth_gtL.squeeze(0)
    #print("new L depth Size:", depth_gtL.size())
if depth_gtR.size() != (375, 1242):
    depth_gtR = resizeT(depth_gtR.unsqueeze(0))
    depth_gtR = depth_gtR.squeeze(0)
    #print("new R depth Size:", depth_gtR.size())

