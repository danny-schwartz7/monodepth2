import torch
from torchvision import transforms
from typing import Tuple
from PIL import Image
import os
import numpy as np
from kitti_utils import generate_depth_map

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
EPSILON = 1e-8

def to_depth(disparity : torch.Tensor, baseline : torch.Tensor, focalLength : torch.Tensor) -> torch.Tensor:
    depth = (baseline.to(DEVICE) * focalLength.to(DEVICE)).unsqueeze(-1)/(disparity+EPSILON) #unsqueeze to make same shape as disparity map
    return depth

def to_disparity(depth : torch.Tensor, baseline : torch.Tensor, focalLength : torch.Tensor) -> torch.Tensor:
    disparity = (baseline.to(DEVICE) * focalLength.to(DEVICE)).unsqueeze(-1)/(depth+EPSILON).to(DEVICE) #unsqueeze to make same shape as disparity map 
    return disparity

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

def get_dataloader(type : str = "train", fraction_of_data_to_use : float = 1.0, batch_size : int = 1, shuffle : bool = False):
    dataset = MyDataset(type, fraction_of_data_to_use)
    loader = torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn = custom_collate, num_workers=4, prefetch_factor = 5)
    return loader

def custom_collate(data):
    out = tuple(map(torch.stack, zip(*data))) #map ith in each tuple to their own tensor
    toReturn = Data_Tuple(imgL=out[0], imgR=out[1], depth_gtL=out[2], depth_gtR=out[3], focalLength=out[4], baseline=out[5])
    return toReturn

class Data_Tuple():
    #basically a struct
    def __init__(self, imgL : torch.Tensor, imgR : torch.Tensor, depth_gtL : torch.Tensor, depth_gtR : torch.Tensor, focalLength : torch.Tensor, baseline : torch.Tensor) -> None:
        self.imgL = imgL
        self.imgR = imgR
        self.depthL = depth_gtL
        self.depthR = depth_gtR
        self.focalLength = focalLength
        self.baseline = baseline

class MyDataset(torch.utils.data.Dataset):
    def __init__(self, type : str, frac_to_use : float = 1.0):
        assert frac_to_use <= 1.0
        self.basedir = 'kitti_data'

        #get images from drives
        allImagesInDrives = self.getAllImagesInDrives(frac_to_use)
        numDrives = len(allImagesInDrives)
        numImages = len([img for drive in allImagesInDrives for img in drive])
        #print("num drives: ", numDrives)
        #print("num images: ", numImages)
        avgImgDrive =  numImages/numDrives
        #print("avg images per drive: ", avgImgDrive)

        splits = [28/30, 1/30, 1/30]
        assert sum(splits) == 1
        #physical number of images
        numTrainImg : int  = int(splits[0]*numImages)
        numTestImg : int   = int(splits[1]*numImages)
        numEvalImg : int   = numImages - numTrainImg - numTestImg
        
        #approx drives
        numTestDrives = int(numTestImg / avgImgDrive)
        numEvalDrives = int(numEvalImg / avgImgDrive)
        if numTestDrives < 1:
            numTestDrives = 1
        if numEvalDrives < 1:
            numEvalDrives = 1
        numTrainDrives = numDrives - numTestDrives - numEvalDrives

        self.dataPathTuples = []     #list of (Limg, Rimg, velo, camDir)
        if type == "train":
            self.dataPathTuples = [img for drive in allImagesInDrives[numTestDrives+numEvalDrives:] for img in drive]
        elif type == "test":
            subset = allImagesInDrives[:numTestDrives]
            self.dataPathTuples = [img for drive in subset for img in drive]
        elif type == "eval":
            self.dataPathTuples = [img for drive in allImagesInDrives[numTestDrives:numEvalDrives+numTestDrives] for img in drive]

        print(f"loaded {len(self.dataPathTuples)} from {numTrainDrives} | {numTestDrives} | {numEvalDrives} images for {type}")

    def getCalibInfo(self, calibDir):
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

    def getAllImagesInDrives(self, frac_to_use : float):
        #path to drive for data
        calibDirs = [f.path  for f in os.scandir(self.basedir) if f.is_dir()]
        
        # print(calibDirs)
        # print("\n", incorrectShapeDrives)
        # calibDirs = list(set(calibDirs) - set(incorrectShapeDrives))
        
        driveFolders = []
        for calibDir in calibDirs:
            driveFolders += [f.path  for f in os.scandir(calibDir) if f.is_dir()]
        
        LcamPath = os.path.join("image_02", "data")
        RcamPath = os.path.join("image_03", "data")
        veloPath = os.path.join("velodyne_points", "data")
        totalImages = []
        for driveFolder in driveFolders:
            driveImages = []
            calibDir = driveFolder.split("/")[1]
            calibDir = os.path.join(self.basedir, calibDir)
            
            #find 02 images
            LImages = sorted([f.path for f in os.scandir(os.path.join(driveFolder, LcamPath))])
            #find 03 images
            RImages = sorted([f.path for f in os.scandir(os.path.join(driveFolder, RcamPath))])
            #find velodyne images
            veloDatas = sorted([f.path for f in os.scandir(os.path.join(driveFolder, veloPath))])

            
            #make tuples with coresponding images
            if not(len(LImages) == len(RImages) and len(LImages) == len(veloDatas)):
                #print("unequal lengths in data fixing errors")
                LFront = LImages[0][:-14]
                RFront = RImages[0][:-14]
                LimageNums = []
                for path in LImages:
                    LimageNums += [path[-14:-4]]
                
                RimageNums = []
                for path in RImages:
                    RimageNums += [path[-14:-4]]
                
                veloNums = []
                for path in veloDatas:
                    veloNums += [path[-14:-4]]
                
                errorFilesL = []
                errorFilesR = []

                if len(veloDatas) < len(LImages) and len(veloDatas) < len(RImages) and len(RImages) == len(LImages):
                    
                    for i, num in enumerate(LimageNums):
                        if num not in veloNums and num in RimageNums:
                            #print(f"fixing error in image/velo corresponding to {num}")
                            errorFilesL += [f"{LFront}{num}.jpg"]
                            errorFilesR += [f"{RFront}{num}.jpg"]
                    
                # for image in LImages:
                #     print(image)
                
                # print("\nerror files: ")
                # for image in errorFilesL:
                #     print(image)

                for L, R in zip(errorFilesL, errorFilesR):
                    LImages.remove(L)
                    RImages.remove(R)

                #print(f"{len(LImages)} : {len(RImages)} : {len(veloDatas)}")
                    
            for i, (Lcam, Rcam, velo) in enumerate(zip(LImages, RImages, veloDatas)):
                #print(f"{i} with {Lcam} and {Rcam} and {velo}")
                if Lcam[-14:-4] == Rcam[-14:-4] and Lcam[-14:-4] == velo[-14:-4] and Rcam[-14:-4] == velo[-14:-4]:
                    #print(f"{driveFolder} with {Lcam[-14:-4]} : {Rcam[-14:-4]} : {velo[-14:-4]}")
                    driveImages += [(Lcam, Rcam, velo, calibDir)]
                else:
                    #print(f"{driveFolder} with {Lcam[-14:-4]} : {Rcam[-14:-4]} : {velo[-14:-4]} error")
                    pass
            
            numImagesInDrive = len(driveImages)
            #print(f"num images in drive: {numImagesInDrive}")
            numToUse = int(numImagesInDrive * frac_to_use)
            if numToUse < 1:
                numToUse = 1
            
            #HACK FOR TESTING
            numToUse = 2
            #print(f"num to use: {len(driveImages[:numToUse])}")

            totalImages += [driveImages[:numToUse]]
        return totalImages

    def __len__(self):
        return len(self.dataPathTuples)

    def __getitem__(self, index) -> tuple: #(torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        cam 2 = left cam color
        cam 3 = right cam color
        """
        (L_imgPath, R_imgPath, veloPath, calibDir) = self.dataPathTuples[index]

        #Get focalLength and baseline
        focalLength, baseline = self.getCalibInfo(calibDir)

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

        #normalize
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        normalizer = transforms.Normalize(mean, std)
        print("imgL mean before", torch.mean(imgL[0]), torch.std(imgL[0]))
        imgL = normalizer(imgL)
        print("imgL mean after", torch.mean(imgL[0]), torch.std(imgL[0]))
        raise
        #retrieve depth data
        depth_gtL = generate_depth_map(calibDir, velo_filename=veloPath, cam = 2)
        depth_gtR = generate_depth_map(calibDir, velo_filename=veloPath, cam = 3)
        
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


        assert imgL.size() == (3, 375, 1242)
        assert imgR.size() == (3, 375, 1242)
        assert depth_gtL.size() == (375, 1242)
        assert depth_gtR.size() == (375, 1242)

        return (imgL, imgR, depth_gtL, depth_gtR, focalLength, baseline)
                
        