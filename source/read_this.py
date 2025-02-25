#!/usr/bin/env python3
import csv


def csv_as_generator(filepath):
    """Return csv content as a generator object of (float, float)."""
    with open(filepath, "r") as f:
        r = csv.reader(f)
        for i in r:
            # Converts to (float, float) tuple before returning
            # File will remain open until the generator is exhausted
            # yield (float(i[0]), float(i[1]))
            yield r


def csv_as_list(filepath):
    """Return csv content as a list of (float, float)."""
    with open(filepath, "r") as f:
        r = csv.reader(f)
        t = list()
        for i in r:
            t.append((float(i[0]), float(i[1])))
        return t
