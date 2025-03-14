#!/usr/bin/env python3
import argparse
from visualizer import consts
from visualizer import utils


def parse_them():
    # Set up for command-line arguments
    parser = argparse.ArgumentParser(description=consts.PROGRAM_DESCRIPTION)

    # Flags
    parser.add_argument(
        "--xkcd",
        help="display the plot in a different style",
        action="store_true",
    )
    parser.add_argument(
        "-d",
        "--dev",
        help="add a cheating-button for quick testing",
        action="store_true",
    )

    # Optional Arguments
    parser.add_argument(
        "-s", "--seed", type=int, help="set seed for all relevant packages"
    )

    args = parser.parse_args()

    # Handle arguments
    consts.flags["xkcd"] = args.xkcd
    consts.flags["dev"] = args.dev

    if args.seed is not None:
        utils.superseed(int(args.seed))
