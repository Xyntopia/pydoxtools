#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 22:12:25 2020
# TODO write file description
"""
from __future__ import annotations

import collections
import datetime
import logging
import math
import numbers
from itertools import groupby
from operator import itemgetter
from typing import List, Tuple, Any, Dict

import pandas as pd

logger = logging.getLogger(__name__)


def isnan(val):
    if isinstance(val, float):
        return math.isnan(val)
    else:
        return False


def flatten(
        list_like,
        drop_none: bool = True,
        max_level: int = -1,
        level: int = 0
):
    """
    flatten everything in args from nested objects into
    "flat" lists
    """
    for el in list_like:
        if (el is not None) and (not isnan(el)):
            if isinstance(el, collections.abc.Iterable) and not isinstance(el, (dict, str, bytes)):
                if level < max_level or max_level == -1:
                    yield from flatten(el, drop_none=drop_none, level=level + 1)
                else:
                    yield el
            elif isinstance(el, (str, bytes)):
                if not el.isspace() and el != "":
                    yield el
            else:
                yield el


def flatten_unique(x):
    return set(flatten(x))


def deep_str_convert(obj):
    """converts all byte strings to str in a nested python object"""
    if isinstance(obj, (list, tuple)):
        res = [deep_str_convert(o) for o in obj]
    elif isinstance(obj, dict):
        res = {deep_str_convert(k): deep_str_convert(v)
               for k, v in obj.items()}
    elif isinstance(obj, str):
        res = obj
    elif isinstance(obj, bytes):
        try:
            res = obj.decode('utf-8')
        except:  # TODO: decode error
            res = str(obj)
    elif isinstance(obj, datetime.datetime):
        res = obj.isoformat(timespec='seconds')
    elif obj is None:
        res = obj
    elif isinstance(obj, pd.DataFrame):
        res = obj.to_dict("records")
        # TODO: recursivly enter the generated dict
    elif isinstance(obj, pd.Series):
        res = obj.to_dict()
    elif isinstance(obj, numbers.Number):
        res = obj  # leave untouched
    else:
        logger.debug(f"can not process object of kind: {type(obj)}"
                     f",\nthe object: {obj}")
        res = str(obj)

    return res


def group_by(data: List[Tuple[str, Any]]) -> Dict[str, Any]:
    """group data given as a list of tuples where the first element of the tuple
    specfies the group key.
    """
    groups = {}
    for k, group in groupby(sorted(data, key=itemgetter(0)), lambda x: x[0]):
        groups[k] = [g[1:] for g in group]

    return groups
