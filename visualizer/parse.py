#!/usr/bin/env python3
import csv


def csv_as_list(filepath):
    """Read csv file and return content as a list of (float, float)."""
    with open(filepath, "r") as f:
        r = csv.reader(f)
        t = list()
        for i in r:
            t.append((float(i[0]), float(i[1])))
        return t
