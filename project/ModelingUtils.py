import torch
from torch import nn
from tqdm import tqdm
from tensorboardX import SummaryWriter
from matplotlib import pyplot as plt
from typing import Optional, Tuple

from project.evaluation import calculate_quantitative_results_RMS, calculate_quantitaive_results_SILog
from project.triangulation import calculateDisparity
from unsupervised.MonodepthUtils import reconstruct_input_from_disp_maps, unsupervised_monodepth_loss, unsupervised_multiscale_monodepth_loss

TRAIN_REPORT_INTERVAL = 50
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def unsupervised_single_scale_loss(tup, model: nn.Module, return_individual_losses: bool = False):
    """

    :param tup: A tuple from the dataloader
    :return: loss
    """

    left_img, right_img, _, _ = tup

    left_img = left_img.to(DEVICE)
    right_img = right_img.to(DEVICE)

    stereo_pair = (left_img, right_img)
    disp_maps = model.forward(left_img)

    reconstructions = reconstruct_input_from_disp_maps(stereo_pair, disp_maps)

    return unsupervised_monodepth_loss(stereo_pair, disp_maps, reconstructions, return_individual_losses=return_individual_losses)


def unsupervised_multi_scale_loss(tup, model: nn.Module, return_individual_losses: bool = False):
    """

    :param tup: A tuple from the dataloader
    :return: loss
    """

    left_img, right_img, _, _ = tup

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

    model = model.to(DEVICE)
    model.train()
    best_val_loss = float("inf")

    tbx_writer = SummaryWriter(log_dir=tbx_log_dir)

    optimizer = torch.optim.Adam(model.parameters(), lr=initial_lr)

    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.95)

    train_tbx_idx = 0
    for epoch in tqdm(range(num_epochs)):
        tbx_writer.add_scalar("lr/lr", lr_scheduler.get_lr()[0], train_tbx_idx)

        num_train_examples = 0
        running_loss = 0
        running_recon_loss = 0
        running_disp_smooth_loss = 0
        running_lr_consistency_loss = 0

        model.train()
        for tup in tqdm(train_loader, desc=f"Training - Epoch {epoch}", leave=False):
            examples_in_batch = tup[0].shape[0]

            optimizer.zero_grad()
            recon_loss, disp_smooth_loss, lr_consistency_loss, total_loss = unsupervised_multi_scale_loss(tup, model, True)
            running_loss += examples_in_batch * total_loss.item()
            running_recon_loss += recon_loss.item()
            running_disp_smooth_loss += disp_smooth_loss.item()
            running_lr_consistency_loss += lr_consistency_loss.item()
            total_loss.backward()
            optimizer.step()

            num_train_examples += examples_in_batch
            train_tbx_idx += examples_in_batch
            if num_train_examples > TRAIN_REPORT_INTERVAL:
                running_loss /= num_train_examples
                tbx_writer.add_scalar("train/loss", running_loss, train_tbx_idx)

                running_recon_loss /= num_train_examples
                tbx_writer.add_scalar("train/reconstruction_loss", running_recon_loss, train_tbx_idx)

                running_disp_smooth_loss /= num_train_examples
                tbx_writer.add_scalar("train/disparity_smoothness_loss", running_disp_smooth_loss, train_tbx_idx)

                running_lr_consistency_loss /= num_train_examples
                tbx_writer.add_scalar("train/lr_consistency_loss", running_lr_consistency_loss, train_tbx_idx)

                running_loss = 0
                running_recon_loss = 0
                running_disp_smooth_loss = 0
                running_lr_consistency_loss = 0
                num_train_examples = 0

        val_loss = 0
        val_recon_loss = 0
        val_disp_smooth_loss = 0
        val_lr_consistency_loss = 0
        total_val_examples = 0
        for tup in tqdm(val_loader, desc=f"Validation - Epoch {epoch}", leave=False):
            model.eval()
            with torch.no_grad():
                examples_in_batch = tup[0].shape[0]

                recon_loss, disp_smooth_loss, lr_consistency_loss, total_loss = unsupervised_multi_scale_loss(tup, model, True)
                val_loss += examples_in_batch * total_loss.item()

                val_recon_loss += recon_loss.item()
                val_disp_smooth_loss += disp_smooth_loss.item()
                val_lr_consistency_loss += lr_consistency_loss.item()

                total_val_examples += examples_in_batch
        val_loss /= total_val_examples
        tbx_writer.add_scalar("val/loss", val_loss, epoch)

        val_recon_loss /= total_val_examples
        tbx_writer.add_scalar("val/reconstruction_loss", val_recon_loss, epoch)

        val_disp_smooth_loss /= total_val_examples
        tbx_writer.add_scalar("val/disparity_smoothness_loss", val_disp_smooth_loss, epoch)

        val_lr_consistency_loss /= total_val_examples
        tbx_writer.add_scalar("val/lr_consistency_loss", val_lr_consistency_loss, epoch)

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

    running_mse_cv = 0
    running_silog_cv = 0
    for tup in tqdm(test_loader):
        with torch.no_grad():
            examples_in_batch = tup[0].shape[0]
            #test_loss += examples_in_batch * unsupervised_multi_scale_loss(tup, model).item()

            gt_left_depth = tup[2]

            out = model.forward(tup[0].to(DEVICE))
            disp_map = out[-1][0]

            for i in range(examples_in_batch):
                cur_left_disp = disp_map[i, :, :]
                cur_depth_gt = gt_left_depth[i, :, :]
                running_mse += calculate_quantitative_results_RMS(cur_left_disp, cur_depth_gt) ** 2
                running_silog += calculate_quantitaive_results_SILog(cur_left_disp, cur_depth_gt)

                imgL, imgR, depth_gtL, depth_gtR = tup[0][i, :, :, :], tup[1][i, :, :, :], tup[2][i, :, :], tup[3][i, :, :]

                disp_cv = torch.tensor(calculateDisparity((imgL, imgR, depth_gtL, depth_gtR)))
                running_mse_cv += calculate_quantitative_results_RMS(disp_cv, cur_depth_gt) ** 2
                running_silog_cv += calculate_quantitaive_results_SILog(disp_cv, cur_depth_gt)

            total_test_examples += examples_in_batch
    # print(f"Average test loss was {test_loss/total_test_examples}")
    print(f"Average model MSE was {running_mse / total_test_examples}")
    print(f"Average model SILog error was {running_silog / total_test_examples}")
    print(f"Average opencv MSE was {running_mse_cv / total_test_examples}")
    print(f"Average opencv SILog error was {running_silog_cv / total_test_examples}")
    # TODO: add a function to compare this to ground-truth depth using methods to convert from disparity to depth

# TODO: add a function that can display generated disp maps for images next to GT depth maps and the original images
def visualize_disparity_maps(data_loader: torch.utils.data.DataLoader, model: nn.Module, savepath: str):
    model.to(DEVICE)

    for tup in data_loader:
        data_tuple_to_plt_image(tup, model)
        plt.savefig(savepath)
        break  # TODO: refactor this to avoid this hacky logic to get access to data


def data_tuple_to_plt_image(tup, model: nn.Module):
    model.eval()

    tup = convert_tuple_to_batched_if_necessary(tup)
    left_image, right_image, left_depth_gt, right_depth_gt = tup
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

    fig.add_subplot(rows, cols, 2)
    plt.imshow(left_depth_gt_np)  # TODO: use cmap?
    plt.axis('off')
    plt.title("Left Ground-Truth Depth")

    fig.add_subplot(rows, cols, 3)
    plt.imshow(left_recon_np)
    plt.axis('off')
    plt.title("Reconstructed Left Image")

    fig.add_subplot(rows, cols, 4)
    plt.imshow(left_disp_np)  # TODO: use cmap?
    plt.axis('off')
    plt.title("Predicted Disparity Map")

    return fig


def convert_tuple_to_batched_if_necessary(tup):
    left_image, right_image, left_depth_gt, right_depth_gt = tup
    if len(left_image.shape) == 4:
        return tup
    left_image = left_image.unsqueeze(dim=0)
    right_image = right_image.unsqueeze(dim=0)
    left_depth_gt = left_depth_gt.unsqueeze(dim=0)
    right_depth_gt = right_depth_gt.unsqueeze(dim=0)
    return (left_image, right_image, left_depth_gt, right_depth_gt)
