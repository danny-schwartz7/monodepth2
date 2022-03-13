import numpy as np
import torch
from scipy.interpolate import LinearNDInterpolator
from torch import nn
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from typing import Optional, Tuple

from project.RunningLossTracker import RunningLossTracker
from project.evaluation import calculate_quantitative_results_RMS, calculate_quantitaive_results_SILog
import dataset_interface
from dataset_interface import Data_Tuple
from unsupervised.MonodepthUtils import reconstruct_input_from_disp_maps, unsupervised_monodepth_loss, unsupervised_multiscale_monodepth_loss

TRAIN_REPORT_INTERVAL = 50
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


SUPERVISED_LOSS_WEIGHT = 1e-1
SEPT_28_FOCAL_LEN = 0.7704891562461853


def mono_semisupervised_MSE_loss(tup: Data_Tuple, model: nn.Module):
    # only use loss for examples where focal_length == 0.7705, else loss component is 0

    left_img = tup.imgL
    left_img = left_img.to(DEVICE)
    disp_maps = model.forward(left_img)
    leftDisp = disp_maps[-1][0]

    gtDepth = tup.depthL.to(DEVICE)
    leftDepth = dataset_interface.to_depth(leftDisp, tup.baseline, tup.focalLength).to(DEVICE)

    se_batched = torch.pow(leftDepth - gtDepth, 2)

    depth_mask = torch.where(gtDepth != 0, torch.ones_like(gtDepth).to(DEVICE), torch.zeros_like(gtDepth).to(DEVICE)).to(DEVICE)
    se_batched = se_batched * depth_mask
    mse_batched = torch.sum(se_batched, dim=(1, 2))/torch.sum(depth_mask, dim=(1, 2))

    supervision_mask = torch.where(tup.focalLength.reshape((-1,)).to(DEVICE) == SEPT_28_FOCAL_LEN,
                                   torch.ones_like(mse_batched).to(DEVICE), torch.zeros_like(mse_batched).to(DEVICE)).to(DEVICE)
    mse_batched = mse_batched * supervision_mask

    if torch.sum(supervision_mask) == 0:
        # avoid dividing by 0
        return torch.sum(supervision_mask)
    mse_loss = torch.sum(mse_batched)/torch.sum(supervision_mask)
    return mse_loss


def unsupervised_single_scale_loss(tup: Data_Tuple, model: nn.Module, return_individual_losses: bool = False):
    """

    :param tup: A tuple from the dataloader
    :return: loss
    """

    left_img, right_img = tup.imgL, tup.imgR

    left_img = left_img.to(DEVICE)
    right_img = right_img.to(DEVICE)

    stereo_pair = (left_img, right_img)
    disp_maps = model.forward(left_img)

    reconstructions = reconstruct_input_from_disp_maps(stereo_pair, disp_maps)

    return unsupervised_monodepth_loss(stereo_pair, disp_maps, reconstructions, return_individual_losses=return_individual_losses)


def unsupervised_multi_scale_loss(tup: Data_Tuple, model: nn.Module, return_individual_losses: bool = False):
    """

    :param tup: A tuple from the dataloader
    :return: loss
    """

    left_img, right_img = tup.imgL, tup.imgR

    left_img = left_img.to(DEVICE)
    right_img = right_img.to(DEVICE)

    stereo_pair = (left_img, right_img)
    disp_maps = model.forward(left_img)

    return unsupervised_multiscale_monodepth_loss(stereo_pair, disp_maps, return_individual_losses=return_individual_losses)


def train(train_loader: torch.utils.data.DataLoader,
            val_loader: torch.utils.data.DataLoader,
            model: nn.Module,
            model_savedir: str,
            tbx_log_dir: str,
            initial_lr: float = 1e-4,
            num_epochs: int = 2,
            supervised: bool = False,
            train_viz_tup: Optional[Tuple] = None,
            val_viz_tup: Optional[Tuple] = None):

    if supervised:
        raise NotImplementedError("Implement this later!")

    assert train_viz_tup[5] != 0.7705, "Train viz tup is from supervised drive (9/28), it shouldn't be"

    model = model.to(DEVICE)
    model.train()
    best_val_loss = float("inf")

    tbx_writer = SummaryWriter(log_dir=tbx_log_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

    def train_ingestor(loss_dict):
        out_dict = {}
        for key in loss_dict.keys():
            out_dict[f"train/{key}"] = loss_dict[key]
        return out_dict

    train_tracker = RunningLossTracker(
        tbx_writer,
        ingestor=train_ingestor
    )

    train_tbx_idx = 0
    for epoch in tqdm(range(num_epochs)):
        tbx_writer.add_scalar("lr/lr", lr_scheduler.get_lr()[0], train_tbx_idx)

        num_train_examples = 0
        num_supervision_batches = 0
        running_supervision_loss = 0

        model.train()
        for tup in tqdm(train_loader, desc=f"Training - Epoch {epoch}", leave=False):
            examples_in_batch = tup.imgL.shape[0]

            optimizer.zero_grad()
            #recon_loss, disp_smooth_loss, lr_consistency_loss, total_loss = unsupervised_multi_scale_loss(tup, model, True)
            unsup_loss_dict = unsupervised_multi_scale_loss(tup, model, True)
            total_loss = unsup_loss_dict["loss"]

            train_tracker.ingest(unsup_loss_dict, examples_in_batch, examples_in_batch)

            # semisup_loss = mono_semisupervised_MSE_loss(tup, model)

            # if semisup_loss != 0:
            #     running_supervision_loss += semisup_loss.item()
            #     num_supervision_batches += 1

            # total_loss += SUPERVISED_LOSS_WEIGHT * semisup_loss

            total_loss.backward()
            optimizer.step()

            num_train_examples += examples_in_batch
            train_tbx_idx += examples_in_batch
            if num_train_examples > TRAIN_REPORT_INTERVAL:
                train_tracker.log(train_tbx_idx)

                if num_supervision_batches > 0:
                    running_supervision_loss /= num_supervision_batches
                    tbx_writer.add_scalar("train/supervised_loss_component", running_supervision_loss, train_tbx_idx)

                running_supervision_loss = 0
                num_train_examples = 0
                num_supervision_batches = 0

        val_loss = 0
        total_val_examples = 0

        def val_ingest(loss_dict):
            out_dict = {}

            for key in loss_dict.keys():
                out_dict[f"val/{key}"] = loss_dict[key]

            return out_dict

        val_tracker = RunningLossTracker(tbx_writer, val_ingest)

        for tup in tqdm(val_loader, desc=f"Validation - Epoch {epoch}", leave=False):
            model.eval()
            with torch.no_grad():
                examples_in_batch = tup.imgL.shape[0]

                unsup_loss_dict = unsupervised_multi_scale_loss(tup, model, True)
                val_loss += examples_in_batch * unsup_loss_dict["loss"].item()

                # don't use supervision in validation set?
                val_tracker.ingest(unsup_loss_dict, examples_in_batch, examples_in_batch)

                total_val_examples += examples_in_batch
        val_tracker.log(epoch)

        val_loss /= total_val_examples

        torch.save(model, f"{model_savedir}/last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, f"{model_savedir}/best.pt")

        # visualize examples
        if train_viz_tup is not None:
            with torch.no_grad():
                fig = data_tuple_to_plt_image(train_viz_tup, model)
            tbx_writer.add_figure("train/visualization", fig, epoch)
            plt.close(fig)
        if val_viz_tup is not None:
            with torch.no_grad():
                fig = data_tuple_to_plt_image(val_viz_tup, model)
            tbx_writer.add_figure("val/visualization", fig, epoch)
            plt.close(fig)

        lr_scheduler.step()

    print(f"Training completed! Best validation loss was {best_val_loss}")


def test(test_loader: torch.utils.data.DataLoader, model: nn.Module):
    model = model.to(DEVICE)
    model.eval()

    test_loss = 0
    total_test_examples = 0
    running_mse = 0
    running_silog = 0

    for tup in tqdm(test_loader):
        with torch.no_grad():
            examples_in_batch = tup.imgL.shape[0]
            #test_loss += examples_in_batch * unsupervised_multi_scale_loss(tup, model).item()

            out = model.forward(tup.imgL.to(DEVICE))
            disp_map = out[-1][0]

            for i in range(examples_in_batch):
                cur_left_disp = disp_map[i, :, :]

                predicted_depth = dataset_interface.to_depth(cur_left_disp, focalLength=tup.focalLength[i],
                                                             baseline=tup.baseline[i])

                individual_tup = (tup.imgL[i, :, :, :], tup.imgR[i, :, :, :], tup.depthL[i, :, :],
                                  tup.depthR[:, :, :], tup.focalLength[i], tup.baseline[i])

                running_mse += calculate_quantitative_results_RMS(predicted_depth, individual_tup) ** 2
                running_silog += calculate_quantitaive_results_SILog(predicted_depth, individual_tup)
            total_test_examples += examples_in_batch
    # print(f"Average test loss was {test_loss/total_test_examples}")
    print(f"Average model MSE was {running_mse / total_test_examples}")
    print(f"Average model SILog error was {running_silog / total_test_examples}")
    # TODO: add a function to compare this to ground-truth depth using methods to convert from disparity to depth

# TODO: add a function that can display generated disp maps for images next to GT depth maps and the original images
def visualize_disparity_maps(data_loader: torch.utils.data.DataLoader, model: nn.Module, savepath: str):
    model.to(DEVICE)

    for tup in data_loader:
        data_tuple_to_plt_image(tup, model)
        plt.savefig(savepath)
        break  # TODO: refactor this to avoid this hacky logic to get access to data


def data_tuple_to_plt_image(tup, model: nn.Module):
    """
    This function still uses the old (length-6 tuple) format based on quirks of Parker's implementation
    """
    model.eval()

    tup = convert_tuple_to_batched_if_necessary(tup)
    left_image, right_image, left_depth_gt, right_depth_gt, focal_length, baseline = tup
    left_image = left_image.to(DEVICE)
    right_image = right_image.to(DEVICE)

    with torch.no_grad():
        left_to_right_disp, right_to_left_disp = model.forward(left_image)[-1]

        recons = reconstruct_input_from_disp_maps((left_image, right_image), (left_to_right_disp, right_to_left_disp))

    left_image_np = left_image.permute((0, 2, 3, 1))[0, :, :, :].cpu().detach().numpy()
    left_disp_np = left_to_right_disp[0, :, :].cpu().detach().numpy()
    left_depth_gt_np = left_depth_gt[0, :, :].cpu().detach().numpy()
    left_recon_np = recons[0].permute((0, 2, 3, 1))[0, :, :, :].cpu().detach().numpy()

    fig = plt.figure(figsize=(21, 7))

    rows = 2
    cols = 2

    fig.add_subplot(rows, cols, 1)
    plt.imshow(left_image_np)
    plt.axis('off')
    plt.title("Left Image")

    x, y = np.where(left_depth_gt_np > 0)
    d = left_depth_gt_np[left_depth_gt_np != 0]
    xyd = np.stack((y, x, d)).T
    gt = lin_interp(left_depth_gt_np.shape, xyd)

    fig.add_subplot(rows, cols, 2)
    plt.imshow(gt)  # TODO: use cmap?
    plt.axis('off')
    plt.set_cmap('plasma')
    plt.title("Left Ground-Truth Depth")

    fig.add_subplot(rows, cols, 3)
    plt.imshow(left_recon_np)
    plt.axis('off')
    plt.title("Reconstructed Left Image")

    left_depth_calc = dataset_interface.to_depth(left_to_right_disp[0, :, :], baseline, focal_length).cpu().detach().numpy()[0,:,:]

    fig.add_subplot(rows, cols, 4)
    plt.imshow(left_depth_calc)
    #plt.imshow(left_disp_np, vmin=0.0, vmax=0.3)  # TODO: use cmap?
    plt.axis('off')
    plt.set_cmap('plasma')
    plt.title("Predicted Depth Map")

    return fig


def convert_tuple_to_batched_if_necessary(tup):
    left_image, right_image, left_depth_gt, right_depth_gt, focal_len, baseline = tup
    if len(left_image.shape) == 4:
        return tup
    left_image = left_image.unsqueeze(dim=0)
    right_image = right_image.unsqueeze(dim=0)
    left_depth_gt = left_depth_gt.unsqueeze(dim=0)
    right_depth_gt = right_depth_gt.unsqueeze(dim=0)
    focal_len = focal_len.unsqueeze(dim=0)
    baseline = baseline.unsqueeze(dim=0)
    return (left_image, right_image, left_depth_gt, right_depth_gt, focal_len, baseline)

def lin_interp(shape, xyd):
    # taken from https://github.com/hunse/kitti
    m, n = shape
    ij, d = xyd[:, 1::-1], xyd[:, 2]
    f = LinearNDInterpolator(ij, d, fill_value=0)
    J, I = np.meshgrid(np.arange(n), np.arange(m))
    IJ = np.vstack([I.flatten(), J.flatten()]).T
    disparity = f(IJ).reshape(shape)
    return disparity
