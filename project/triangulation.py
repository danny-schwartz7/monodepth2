import argparse
import os
from torch.utils.data import DataLoader
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import torch
import time
#import dataset_interface

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
    imgL = np.random.randint(low=5,high=255,size=(3,375,1242)).astype(np.uint8)
    imgR = np.random.randint(low=5,high=255,size=(3,375,1242)).astype(np.uint8)

    imgLGray = 0.2989 * imgL[0,:,:] + 0.5870 * imgL[1,:,:] + 0.1140 * imgL[2,:,:]
    imgRGray = 0.2989 * imgR[0, :, :] + 0.5870 * imgR[1, :, :] + 0.1140 * imgR[2, :, :]

    stereo = cv.StereoBM_create(numDisparities=16, blockSize=15)
    disparity = stereo.compute(imgLGray.astype(np.uint8), imgRGray.astype(np.uint8))

    return disparity

def calculateDisparity(tup):
    imgL, imgR, depth_gtL, depth_gtR = tup
    imgL = imgL.cpu().detach().numpy()
    imgR = imgR.cpu().detach().numpy()
    depth_gtL = depth_gtL.cpu().detach().numpy()
    depth_gtR = depth_gtR.cpu().detach().numpy()

    imgLGray = 0.2989 * imgL[0, :, :] + 0.5870 * imgL[1, :, :] + 0.1140 * imgL[2, :, :]
    imgRGray = 0.2989 * imgR[0, :, :] + 0.5870 * imgR[1, :, :] + 0.1140 * imgR[2, :, :]

    imgLGray = imgLGray * 255
    imgRGray = imgRGray * 255

    num_disp = int(0.3*1242)
    num_disp -= num_disp % 16

    stereo = cv.StereoBM_create(numDisparities=num_disp, blockSize=15)
    disparity = stereo.compute(imgLGray.astype(np.uint8), imgRGray.astype(np.uint8))

    return disparity

def convertDisparityArrayToTensor(disparity) -> torch.tensor:
    return torch.tensor(disparity)


def data_tuple_to_plt_image(tup):

    left_image, right_image, left_depth_gt, right_depth_gt = tup
    disparity = calculateDisparity(tup)
    dataset = MyDataset("train")
    depth = dataset_interface.to_depth(torch.tensor(disparity), torch.tensor(dataset.baseline), torch.tensor(dataset.focalLength)).cpu().detach().numpy()

    left_image_np = left_image.permute((1, 2, 0)).cpu().detach().numpy()
    right_image_np = right_image.permute((1, 2, 0)).cpu().detach().numpy()
    left_depth_gt_np = left_depth_gt.cpu().detach().numpy()
    right_depth_gt_np = right_depth_gt.cpu().detach().numpy()

    fig = plt.figure(figsize=(21, 7))

    rows = 3
    cols = 2

    fig.add_subplot(rows, cols, 1)
    plt.imshow(left_image_np)
    plt.axis('off')
    plt.title("Left Image")

    fig.add_subplot(rows, cols, 2)
    plt.imshow(left_depth_gt_np)  # TODO: use cmap?
    plt.axis('off')
    plt.title("Left Ground-Truth Depth")

    fig.add_subplot(rows, cols, 3)
    plt.imshow(right_image_np)
    plt.axis('off')
    plt.title("Right Image")

    fig.add_subplot(rows, cols, 4)
    plt.imshow(right_depth_gt_np)  # TODO: use cmap?
    plt.axis('off')
    plt.title("Right Ground-Truth Depth")

    fig.add_subplot(rows, cols, 5)
    plt.imshow(disparity)
    plt.axis('off')
    plt.title("Predicted Disparity Map")

    fig.add_subplot(rows, cols, 6)
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

        plot = True
        evaluate = True
        #Save file
        for filename, loader in [("train", training_dataset), ("val", eval_dataset)]:
            if plot:
                for tup in loader:
                    fig = data_tuple_to_plt_image(tup)
                    plt.savefig(f"/home/alexeve3967/{filename}.png") #Edit depending on which directory you want
                    plt.close(fig)
                    break
            if evaluate:
                running_mse = 0
                running_silog = 0
                n = 0
                for tup in loader:
                    imgL, imgR, depth_gtL, depth_gtR = tup
                    disp_cv = torch.tensor(calculateDisparity(tup))
                    running_mse += calculate_quantitative_results_RMS(disp_cv, depth_gtL) ** 2
                    running_silog += calculate_quantitaive_results_SILog(disp_cv, depth_gtL)
                    n = n + 1
                print("MSE average is ", running_mse)
                print("SILog average is ", running_silog)

def mainTest():
    start_time = time.time()
    disparity = calculateDisparityTest()
    print("--- %s seconds ---" % (time.time() - start_time))
    plt.imshow(disparity, 'gray')
    plt.show()

if __name__ == "__main__":
    main()

