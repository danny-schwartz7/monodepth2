import argparse
import os
from torch.utils.data import DataLoader
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import torch
import time
import dataset_interface
import pickle
from scipy.interpolate import griddata, LinearNDInterpolator
from tqdm import tqdm


from project.evaluation import calculate_quantitative_results_RMS, calculate_quantitaive_results_SILog

from dataset_interface import MyDataset


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--num-epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=4)

    parser.add_argument('--visualize-dir', type=str,
                        help="location of directory to save visualized results to")

    return parser.parse_args()


def make_dir_if_not_exists(dir: str):
    if not os.path.exists(dir):
        os.mkdir(dir)

def calculateDisparityTest():
    # Size of images is 3 x 375 x 1242

    #imgL = np.random.randint(low=5,high=255,size=(3,375,1242)).astype(np.uint8)
    #imgR = np.random.randint(low=5,high=255,size=(3,375,1242)).astype(np.uint8)

    with open('data_tup.pkl', 'rb') as f:
        data_tup_batch = pickle.load(f)

    tup = data_tup_batch
    imgL, imgR, depth_gtL, focal_length, baseline = tup.imgL, tup.imgR, tup.depthL, tup.focalLength, tup.baseline
    imgL = imgL[1, :, :, :].cpu().detach().numpy()  # Good numbers 1, 2?, 3, 6
    imgR = imgR[1, :, :, :].cpu().detach().numpy()
    one_ground_truth_depth = depth_gtL[1, :, :]

    left_image_to_plot = np.transpose(imgL,(1, 2, 0))
    right_image_to_plot = np.transpose(imgR,(1, 2, 0))

    imgLGray = 0.2989 * imgL[0,:,:] + 0.5870 * imgL[1,:,:] + 0.1140 * imgL[2,:,:]
    imgRGray = 0.2989 * imgR[0, :, :] + 0.5870 * imgR[1, :, :] + 0.1140 * imgR[2, :, :]

    print("Left image width is ", imgLGray.shape[1])

    imgLGray = imgLGray * 255
    imgRGray = imgRGray * 255

    num_disp = int(0.3 * 1242)
    num_disp -= num_disp % 16
    num_disp = 368

    stereo = cv.StereoBM_create(numDisparities=num_disp, blockSize=15)
    disparity = stereo.compute(imgLGray.astype(np.uint8), imgRGray.astype(np.uint8))

    fig = plt.figure(figsize=(21, 7))

    rows = 2
    cols = 2

    fig.add_subplot(rows, cols, 1)
    plt.imshow(left_image_to_plot)
    plt.axis('off')
    plt.title("Left Image")

    fig.add_subplot(rows, cols, 2)
    plt.imshow(right_image_to_plot)
    plt.axis('off')
    plt.title("Right Image")

    fig.add_subplot(rows, cols, 4)
    plt.imshow(disparity)
    plt.axis('off')
    plt.set_cmap('plasma')
    plt.title("Disparity")

    disparity = (disparity / imgLGray.shape[1] / 16)
    disparity[disparity < 0] = 0

    print("Max disparity is ", disparity.max())
    print("Min disparity is ", disparity.min())

    print(one_ground_truth_depth.shape)

    depth_np = one_ground_truth_depth.cpu().detach().numpy()

    x, y = np.where(depth_np > 0)
    d = depth_np[depth_np != 0]
    xyd = np.stack((y, x, d)).T
    gt = lin_interp(depth_np.shape, xyd)

    fig.add_subplot(rows, cols, 3)
    plt.imshow(gt)  # TODO: use cmap?
    plt.axis('off')
    plt.set_cmap('plasma')
    plt.title("Left Ground-Truth Depth")
    plt.show()

    return disparity

def calculateDisparity(tup):
    imgL, imgR, depth_gtL, focal_length, baseline = tup.imgL, tup.imgR, tup.depthL, tup.focalLength, tup.baseline
    imgL = imgL.cpu().detach().numpy()
    imgR = imgR.cpu().detach().numpy()
    depth_gtL = depth_gtL.cpu().detach().numpy()

    imgLGray = 0.2989 * imgL[0, :, :] + 0.5870 * imgL[1, :, :] + 0.1140 * imgL[2, :, :]
    imgRGray = 0.2989 * imgR[0, :, :] + 0.5870 * imgR[1, :, :] + 0.1140 * imgR[2, :, :]

    imgLGray = imgLGray * 255
    imgRGray = imgRGray * 255

    num_disp = int(0.3*1242)
    num_disp -= num_disp % 16

    stereo = cv.StereoBM_create(numDisparities=num_disp, blockSize=15)
    disparity = stereo.compute(imgLGray.astype(np.uint8), imgRGray.astype(np.uint8))

    disparity = (disparity / imgLGray.shape[1] / 16)
    disparity[disparity < 0] = 0

    return disparity

def convertDisparityArrayToTensor(disparity) -> torch.tensor:
    return torch.tensor(disparity)


def data_tuple_to_plt_image(tup):

    left_image, right_image, left_depth_gt, right_depth_gt, focal_length, baseline = tup
    disparity = calculateDisparity(tup)
    dataset = MyDataset("train")
    depth = dataset_interface.to_depth(torch.tensor(disparity), torch.tensor(dataset.baseline), torch.tensor(dataset.focalLength)).cpu().detach().numpy()

    left_image_np = left_image.permute((1, 2, 0)).cpu().detach().numpy()
    right_image_np = right_image.permute((1, 2, 0)).cpu().detach().numpy()
    left_depth_gt_np = left_depth_gt.cpu().detach().numpy()
    right_depth_gt_np = right_depth_gt.cpu().detach().numpy()

    fig = plt.figure(figsize=(21, 7))

    rows = 2
    cols = 2

    fig.add_subplot(rows, cols, 1)
    plt.imshow(left_image_np)
    plt.axis('off')
    plt.title("Left Image")

    left_depth_gt_nonzero = left_depth_gt_np[left_depth_gt_np.nonzero(as_tuple=True)]
    left_depth_gt_np_mean = np.mean(left_depth_gt_nonzero)
    left_depth_gt_np = left_depth_gt_np + left_depth_gt_np - left_depth_gt_np_mean
    left_depth_gt_np[(left_depth_gt_np < 0)] = 0
    left_depth_gt_np[(left_depth_gt_np > 255)] = 255

    fig.add_subplot(rows, cols, 2)
    plt.imshow(left_depth_gt_np)  # TODO: use cmap?
    plt.axis('off')
    plt.title("Left Ground-Truth Depth")

    #fig.add_subplot(rows, cols, 3)
    #plt.imshow(right_image_np)
    #plt.axis('off')
    #plt.title("Right Image")

    #right_depth_gt_nonzero = right_depth_gt_np[right_depth_gt_np.nonzero(as_tuple=True)]
    #right_depth_gt_np_mean = np.mean(right_depth_gt_nonzero)
    #right_depth_gt_np = right_depth_gt_np + right_depth_gt_np - left_depth_gt_np_mean
    #right_depth_gt_np[(right_depth_gt_np < 0)] = 0
    #right_depth_gt_np[(right_depth_gt_np > 255)] = 255

    #fig.add_subplot(rows, cols, 4)
    #plt.imshow(right_depth_gt_np)  # TODO: use cmap?
    #plt.axis('off')
    #plt.title("Right Ground-Truth Depth")

    fig.add_subplot(rows, cols, 3)
    plt.imshow(disparity, vmin=0.0, vmax=0.3)
    plt.axis('off')
    plt.title("Predicted Disparity Map")

    fig.add_subplot(rows, cols, 4)
    plt.imshow(depth)  # TODO: use cmap?
    plt.axis('off')
    plt.title("Predicted Depth Map")

    return fig


def main():
        # Size of images is 375 x 1242
        args = get_args()

        training_dataset = MyDataset("train")
        eval_dataset = MyDataset("eval")
        train_loader = DataLoader(dataset=training_dataset, batch_size=args.batch_size)
        val_loader = DataLoader(dataset=eval_dataset, batch_size=args.batch_size)
        test_loader = DataLoader(dataset=MyDataset("test"), batch_size=args.batch_size)

        plot = False
        evaluate = True
        #Save file
        for filename, loader in [("train", training_dataset), ("val", eval_dataset)]:
            if plot:
                for tup in tqdm(loader):
                    fig = data_tuple_to_plt_image(tup)
                    plt.savefig(f"/home/alexeve3967/{filename}.png") #Edit depending on which directory you want
                    plt.close(fig)
                    break
            if evaluate:
                running_mse = 0
                running_silog = 0
                n = 0
                for tup in tqdm(loader):
                    disp_cv = torch.tensor(calculateDisparity(tup))
                    running_mse += calculate_quantitative_results_RMS(disp_cv, tup) ** 2
                    running_silog += calculate_quantitaive_results_SILog(disp_cv, tup)
                    n = n + 1
                print("MSE average is ", running_mse / n)
                print("SILog average is ", running_silog / n)

def mainTest():
    start_time = time.time()
    disparity = calculateDisparityTest()
    print("--- %s seconds ---" % (time.time() - start_time))
    #plt.imshow(disparity, 'gray')
    #plt.show()

def lin_interp(shape, xyd):
    # taken from https://github.com/hunse/kitti
    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity

def use_pickle_files():

    with open('data_tup.pkl', 'rb') as f:
        data_tup_batch = pickle.load(f)

    with open('disp_maps.pkl', 'rb') as f2:
        disp_maps_batch = pickle.load(f2)

    left_to_right_disparity, right_to_left_disparity = disp_maps_batch
    tup = data_tup_batch
    imgL, imgR, depth_gtL, focal_length, baseline = tup.imgL, tup.imgR, tup.depthL, tup.focalLength, tup.baseline

    left_image_np = imgL.permute((0, 2, 3, 1))[0, :, :, :].cpu().detach().numpy() #Good numbers 1, 2?, 3, 6

    one_ground_truth_depth = depth_gtL[0,:,:]
    print(one_ground_truth_depth.shape)

    fig = plt.figure(figsize=(21, 7))

    print("Max is ", torch.max(left_to_right_disparity))
    print("Min is ", torch.min(left_to_right_disparity))

    rows = 2
    cols = 2

    fig.add_subplot(rows, cols, 1)
    plt.imshow(one_ground_truth_depth)  # TODO: use cmap?
    plt.axis('off')
    plt.title("Left Ground-Truth Depth")

    depth_np = one_ground_truth_depth.cpu().detach().numpy()

    x, y = np.where(depth_np > 0)
    d = depth_np[depth_np != 0]
    xyd = np.stack((y, x, d)).T
    gt = lin_interp(depth_np.shape, xyd)

    fig.add_subplot(rows, cols, 3)
    plt.imshow(gt)  # TODO: use cmap?
    plt.axis('off')
    plt.set_cmap('plasma')
    plt.title("Left Ground-Truth Depth")

    fig.add_subplot(rows, cols, 4)
    plt.imshow(left_image_np)  # TODO: use cmap?
    plt.axis('off')
    plt.title("Left Ground-Truth Depth")


    plt.show()

if __name__ == "__main__":
    mainTest()
