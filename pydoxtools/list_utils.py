#!/usr/bin/env python3
"""
Created on Thu Apr  9 22:12:25 2020
# TODO write file description
"""
from __future__ import annotations

import collections
import dataclasses
import datetime
import logging
import math
import numbers
from collections.abc import Iterable, Mapping, MutableMapping
from itertools import groupby
from operator import itemgetter
from typing import Any, Dict

import pandas as pd
import pydantic

logger = logging.getLogger(__name__)


def ensure_list(property):
    """
    This function turns everything into an iterable
    if somthing is a string, it will be turned into list[str]
    if something is already a list it will stay the same way
    """

    # TODO: we need to define more special cases here...
    if isinstance(property, (str, bytes)):
        return [property]
    elif isinstance(property, Iterable):
        return property
    elif property is None:
        return []
    else:
        return [property]


def isnan(val):
    if isinstance(val, float):
        return math.isnan(val)
    else:
        return False


def flatten(
        list_like: Iterable[Any],
        drop_none: bool = True,
        max_level: int = -1,
        level: int = 0
):
    """
    Flattens a nested iterable structure (e.g., list or tuple) into a single-level generator.

    This function recursively traverses through the input iterable structure and yields elements
    at each level. The traversal stops at the specified maximum depth (max_level). The function
    supports skipping None values and can also handle strings and bytes as input.

    Args:
        list_like (Iterable): The input iterable structure (e.g., list, tuple) to be flattened.
        drop_none (bool, optional): If True, None values will be skipped during flattening.
                                     Defaults to True.
        max_level (int, optional): The maximum depth to flatten the nested structure. A value of -1
                                   means there's no limit to the depth. Defaults to -1.
        level (int, optional): The current depth level during recursion. This value should not be
                               changed when calling the function. Defaults to 0.

    Yields:
        Generator: A generator that yields the flattened elements from the input iterable structure.

    Example:
        >>> list(flatten([1, [2, [3, 4], 5], 6]))
        [1, 2, 3, 4, 5, 6]
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


def flatten_dict(
        dct: dict[str, Any],
        prefix: str = '',
        separator: str = '.'
) -> dict[str, Any]:
    """
    Flattens a nested dictionary structure into a single-level dictionary.

    This function recursively traverses through the input dictionary structure and creates new keys
    for nested dictionaries using dot notation. The traversal stops at the specified maximum depth.

    Args:
        dct (Dict[str, Any]): The input dictionary structure to be flattened.
        prefix (str, optional): The prefix for the current level's keys. Defaults to ''.
        separator (str, optional): The separator to use between nested keys. Defaults to '.'.

    Returns:
        Dict[str, Any]: A dictionary with the flattened structure from the input nested dictionary.

    Example:
        >>> flatten_dict({'a': {'b': {'c': 1}}, 'd': 2})
        {'a.b.c': 1, 'd': 2}
    """
    result = {}
    for key, value in dct.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        if isinstance(value, MutableMapping):
            result.update(flatten_dict(value, prefix=new_key, separator=separator))
        else:
            result[new_key] = value
    return result


def deep_str_convert(obj: Any) -> Any:
    """
    Recursively process a nested Python object, converting various types to their string representations.

    This function traverses a nested Python object and processes various types of objects, including:
    - Converting byte strings to str using utf-8 encoding
    - Converting datetime objects to ISO 8601 formatted strings
    - Converting pandas DataFrames to dicts with "records" orientation
    - Converting pandas Series to dicts

    The function supports various types of objects, including iterables, mappings, datetime objects,
    pandas DataFrames and Series, and numbers.

    Args:
        obj (Any): The input Python object to be processed. This can be any type of object, such as
                   lists, tuples, sets, dicts, datetime objects, pandas DataFrames or Series, etc.

    Returns:
        Any: The processed Python object with various types of objects converted to their string
             representations or other appropriate formats.

    Raises:
        UnicodeDecodeError: Raised when the byte string cannot be decoded using utf-8 encoding.
                            In this case, the function falls back to converting the byte string to str.

    Example:
        >>> deep_str_convert({b'key': [b'value1', b'value2', {'nested_key': b'nested_value'}]})
        {'key': ['value1', 'value2', {'nested_key': 'nested_value'}]}
    """
    if isinstance(obj, Iterable) and not isinstance(obj, (str, bytes)):
        if isinstance(obj, Mapping):
            res = {deep_str_convert(k): deep_str_convert(v)
                   for k, v in obj.items()}
        else:
            res = [deep_str_convert(o) for o in obj]
    elif isinstance(obj, str):
        res = obj
    elif isinstance(obj, pydantic.BaseModel):
        res = deep_str_convert(obj.dict())
    elif dataclasses.is_dataclass(obj):
        res = deep_str_convert(dataclasses.asdict(obj))
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
        res = deep_str_convert(obj.to_dict())
    elif isinstance(obj, numbers.Number):
        res = obj  # leave untouched
    else:
        logger.debug(f"can not process object of kind: {type(obj)}"
                     f",\nthe object: {obj}")
        res = str(obj)

    return res


def group_by(data: list[tuple[str, Any]]) -> dict[str, Any]:
    """group data given as a list of tuples where the first element of the tuple
    specfies the group key.
    """
    groups = {}
    for k, group in groupby(sorted(data, key=itemgetter(0)), lambda x: x[0]):
        groups[k] = [g[1:] for g in group]

    return groups


def remove_list_from_lonely_object(obj):
    """remove list if obj is a list and only has a single member"""
    try:
        if len(obj) == 1:
            if isinstance(obj, list):
                obj = obj[0]
            elif isinstance(obj, dict):
                obj = next(iter(obj.values()))
    except TypeError:  # if obj doesn't have a "len" attribute
        pass

    return obj
