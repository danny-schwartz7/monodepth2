import argparse
import os
from torch.utils.data import DataLoader

from unsupervised.StereoEncDecNet import StereoEncDecNet
from project.ModelingUtils import train
from dataset_interface import MyDataset, get_dataloader


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-save-dir', type=str, required=True)
    parser.add_argument('--tbx-log-dir', type=str, required=True)
    parser.add_argument('--initial-lr', type=float, default=1e-4)
    parser.add_argument('--num-epochs', type=int, default=2)
    parser.add_argument('--batch-size', type=int, default=20)

    return parser.parse_args()


def make_dir_if_not_exists(dir: str):
    if not os.path.exists(dir):
        os.mkdir(dir)


def main():
    args = get_args()

    make_dir_if_not_exists(args.model_save_dir)
    make_dir_if_not_exists(args.tbx_log_dir)

    model = StereoEncDecNet()

    train_viz_tup = MyDataset("train")[0]
    val_viz_tup = MyDataset("eval")[0]

    train_loader = get_dataloader("train", batch_size=args.batch_size, shuffle=True, fraction_of_data_to_use=0.005)
    val_loader = get_dataloader("eval", batch_size=args.batch_size, shuffle=True, fraction_of_data_to_use=0.005)
    train(train_loader, val_loader, model,
          args.model_save_dir, args.tbx_log_dir,
          args.initial_lr, args.num_epochs, train_viz_tup=train_viz_tup,
          val_viz_tup=val_viz_tup)


if __name__ == "__main__":
    main()
