#!/usr/bin/env python3
import argparse
import time
from visualizer import consts
from visualizer import utils


def parse_them():
    """
    Parse the arguments to be called from CLI.

    Meant to be called as basically the first thing in the program.
    """
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
        # Consider adding torch determinism-only
    else:
        # Just get some value to seed
        # Seeding *may* affect overall torch performance, so let's consider
        # whether to have this determinism-by-principle principle
        utils.superseed(int(time.time()) % 10000)
