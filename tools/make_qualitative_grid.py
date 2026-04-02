import argparse
from math import ceil
from pathlib import Path

from PIL import Image


def main():
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image",
        action="append",
        required=True,
        help="image path to place in the grid; can be passed multiple times",
    )
    parser.add_argument(
        "--title",
        action="append",
        default=[],
        help="optional title for each image, in the same order as --image",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="output figure path",
    )
    parser.add_argument("--ncols", type=int, default=3, help="number of columns")
    parser.add_argument("--dpi", type=int, default=220, help="saved figure DPI")
    args = parser.parse_args()

    image_paths = [Path(path) for path in args.image]
    titles = list(args.title)
    if titles and len(titles) != len(image_paths):
        raise ValueError("The number of --title values must match the number of --image values.")

    ncols = max(1, args.ncols)
    nrows = ceil(len(image_paths) / ncols)
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.5 * ncols, 4.5 * nrows),
        constrained_layout=True,
    )
    if not hasattr(axes, "flat"):
        axes = [axes]
    else:
        axes = list(axes.flat)

    for ax in axes:
        ax.axis("off")

    for idx, image_path in enumerate(image_paths):
        image = Image.open(image_path).convert("RGB")
        axes[idx].imshow(image)
        if titles:
            axes[idx].set_title(titles[idx])
        else:
            axes[idx].set_title(image_path.stem)
        axes[idx].axis("off")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi)
    plt.close(fig)


if __name__ == "__main__":
    main()
