import argparse
import torch
from matplotlib import pyplot as plt

from project.ModelingUtils import test, data_tuple_to_plt_image
from project.train import make_dir_if_not_exists
from dataset_interface import MyDataset, get_dataloader


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model-save-dir', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=20)

    parser.add_argument('--skip-metrics', action="store_true",
                        help="specify this to skip computation of depth performance metrics.")
    parser.add_argument('--visualize-dir', type=str,
                        help="location of directory to save visualized results to")
    parser.add_argument('--num-viz-examples-per-split', type=int, default=5,
                        help="number of examples per dataset split to visualize")

    return parser.parse_args()


def main():
    args = get_args()

    model = torch.load(f"{args.model_save_dir}/best.pt")
    test_loader = get_dataloader("test", batch_size=args.batch_size, shuffle=False)

    if args.skip_metrics is None:
        test(test_loader, model)

    if args.visualize_dir is not None:
        make_dir_if_not_exists(args.visualize_dir)
        for split in ["train", "eval", "test"]:
            dataset = MyDataset(split)
            for i in range(args.num_viz_examples_per_split):
                viz_tup = dataset[i]

                fig = data_tuple_to_plt_image(viz_tup, model)
                save_path = f"{args.visualize_dir}/{split}_{i}.png"
                plt.savefig(save_path)
                plt.close(fig)
            del dataset


if __name__ == "__main__":
    main()