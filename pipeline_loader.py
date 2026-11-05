# -*- coding: utf-8 -*-
"""Expose the thesis module's definitions as a plain namespace dictionary.
"""
import model_panelled_refined_error as _pipeline


def load(path=None):
    """Return {name: object} for everything the thesis module defines."""
    return vars(_pipeline)


if __name__ == "__main__":
    ns = load()
    print("%d names exported by %s" % (len(ns), _pipeline.__name__))
