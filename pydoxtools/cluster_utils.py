"""
Potentially cython-optimizable code for geometric calculations should go here.
"""
from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import difflib
import functools
import typing
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn import cluster as cluster
from sklearn.metrics import pairwise_distances

# which columns in lists describe the bounding box coordinates (for readability/convenience)
# pdf has 0,0: left, bottom
# where x0, y0 left bottom and x1, y1 upper right
x0, y0, x1, y1, x_mean, y_mean = 0, 1, 2, 3, 4, 5
box_cols = ["x0", "y0", "x1", "y1"]  # box columns from a pandas dataframe
box_cols_np = [x0, y0, x1, y1]


@functools.lru_cache(maxsize=256)
def lower_triangle_indices_cached(n: int):
    """
    calculates lower triangle indices which can be used to as indices
    for pairwise function calculations (e.g. pairwise distance matrix).

    cached version of numpys matrix lower-triangle-indices calculation
    function. This makes sense as the function is very easy to cache.
    """
    return np.tril_indices(n, -1)


@functools.lru_cache(maxsize=256)
def linear_pairwise_indices(n: int):
    pair_idx = np.zeros((2, n), dtype=int)
    pair_idx[1] = np.arange(n)
    return pair_idx


def ltri2square(lower_tri, n, tri_idx, diag: np.ndarray = np.nan):
    """convert an array which represents a lower triangle matrix
    into matrix square form"""
    D = np.zeros((n, n))  # create empty square matrix
    D[tri_idx] = lower_tri  # fill triangle
    # np.diag(areas)  # put the matrix back together
    # and put "NaN"s in the diagonal, as it isn't relevant
    di = np.diag_indices(n)
    D = D + D.T
    D[di] = diag
    return D


def pairwise_minbb_areas(boxes: np.ndarray, pair_idx: np.ndarray):
    """
    Calculate the minimum bounding box area for each pair of boxes
    given in "pair_idx".

    minimum boundingbox of b1 and b2
     ┌──────────┬──────────┐
     │          │          │
     │    b1    │          │bb
     │          │  ┌───────┤
     ├──────────┘  │       │
     │             │  b2   │
     │             │       │
     └─────────────┴───────┘

    """
    # calculate all pairwise boundingboxes
    # where p1 is the lower left corner
    # and p2 the upper right
    p1 = np.minimum(boxes[pair_idx[0], :2], boxes[pair_idx[1], :2])
    p2 = np.maximum(boxes[pair_idx[0], 2:], boxes[pair_idx[1], 2:])
    bb_area = np.prod(p2 - p1, 1)
    return bb_area


def individual_box_w_h(boxes: np.ndarray) -> np.ndarray:
    return boxes[:, [x1, y1]] - boxes[:, [x0, y0]]


def individual_box_areas(boxes: np.ndarray) -> np.ndarray:
    return np.prod(individual_box_w_h(boxes), axis=1)


def pairwise_box_area_distance_func(
        boxes, pair_idx,
        allow_negative=False
):
    """
    This function calculates the pairwise distances between two boxes of
    pair_idx using the "left-over-areas" when subtracting the areas of the
    individual boxes from their combined minimum boundingbox.

    it works the following way:

    minimum boundingbox of b1 and b2
     ┌──────────┬──────────┐
     │          │          │
     │    b1    │          │bb
     │          │  ┌───────┤
     ├──────────┘  │       │
     │             │  b2   │
     │d_box_area   │       │
     └─────────────┴───────┘

    The distance is represented by:

        d_box_area = bb_area - b1_area - b2_area

    If d_box_area becomes negative, it means the boxes are definitly overlapping.
    But the boxes could already be overlapping while the d_box_area is not yet
    negative.
    Maximum negativity is the area of the smaller box.

    can be used like this to create a distance matrix:

    >>> calc_pairwise_matrix(pairwise_box_area_distance_func, boxes)
    """
    pairwise_areas = pairwise_minbb_areas(boxes, pair_idx)
    individual_areas = individual_box_areas(boxes)

    # substract area for each box from combined minimum bounding box
    d = pairwise_areas - individual_areas[pair_idx[0]] - individual_areas[pair_idx[1]]
    # set 0 as lower threshold (== boundingboxes touch/overlap)
    if not allow_negative:
        d = np.maximum(d, 0)
    return d


def pairwise_bbox_length_along_axis(boxes: np.ndarray, pair_idx: np.ndarray, axis: int):
    """calculate the length of a boundingbox along specified axis"""

    # calculate height of pairwise boundingboxes along axis
    c0, c1 = (y0, y1) if axis == 1 else (x0, x1)
    c0_min = np.minimum(boxes[pair_idx[0], c0], boxes[pair_idx[1], c0])
    c1_max = np.maximum(boxes[pair_idx[0], c1], boxes[pair_idx[1], c1])

    # length of minimum boundingbox along the axis
    l_along_axis = c1_max - c0_min
    return l_along_axis


def individual_box_length(boxes, axis):
    """calculate length of each individual box along axis"""
    c0, c1 = (y0, y1) if axis == 1 else (x0, x1)
    lb = boxes[:, c1] - boxes[:, c0]
    return lb


def pairwise_box_gap_distance_along_axis_func(boxes, pair_idx, axis):
    """Calculates pairwise distance in one axis direction (0:x, 1:y)
    for a list of boxes

    this also works for lines...

    works like this:
                                   -d
                                 ◄─────►
     ┌────────┐                  ┌────────────┐
     │   b2   │      ┌───────────┼─────┐  b3  │
     └────────┘      │           │     │      │
                     │       b1  └─────┼──────┘
                 d   │                 │
               ◄────►└─────────────────┘
     ────────────►
         axis (x-axis in this example)

    with *d* as the distance between boxes...
    If boxes overlap, *d* becomes negative.
    If *d* is <0, the boxes are overlapping in the direction of this axis
    where the distance in this case represents the length at which the boxes overlap.
    """

    # length of minimum boundingbox along the axis
    l_along_axis = pairwise_bbox_length_along_axis(boxes, pair_idx, axis)
    lb = individual_box_length(boxes, axis)

    # substract lengths of individual boxes from minimum boundingbox
    # not allowing negativ numbers. 0 mean: boundingboxes are touching
    # if one box edge segment is overlapping with the coordinates of the other along
    # the specified axis, the distance becomes negativ
    # the maximum negativ number will be the edge length along the axis
    # of the smaller box and the distance in this case represents the distance
    # that the two boxes are ovelapping
    dy = l_along_axis - lb[pair_idx[0]] - lb[pair_idx[1]]

    return dy


def pairwise_euclidean_distance(data, pair_idx):
    """calculates pairwise euclidean distance"""
    d = data[pair_idx[0], :] - data[pair_idx[1], :]
    d = np.sqrt((d * d).sum(1))
    return d


def pairwise_manhatten_distance(data, pair_idx):
    """calculates pairwise manhatten distance"""
    d = np.abs(data[pair_idx[0], :] - data[pair_idx[1], :]).sum(1)
    return d


def pairwise_string_diff(strdata: str, pair_idx):
    """Calculate pairwise distance between strings using
    python difflib SequenceMatcher"""
    d = [1 - difflib.SequenceMatcher(None, strdata[i_a], strdata[i_b]).ratio()
         for i_a, i_b in zip(*pair_idx)]
    return d


def pairwise_cosine_distance(data, pair_idx):
    """calculates pairwise cosine similarity (distance)"""
    a, b = data[pair_idx[0], :], data[pair_idx[1], :]
    IaI = np.sqrt((a * a).sum(1))
    IbI = np.sqrt((b * b).sum(1))
    ab = (a * b).sum(1)
    d = ab / (IaI * IbI)
    return d


pairwise_l1_distance = pairwise_manhatten_distance
pairwise_l2_distance = pairwise_euclidean_distance


def pairwise_area_size_similarity(boxes, pair_idx):
    """
    calculate similarity between area sizes

    calculates the pairwise ratio of areas.

    return 1.0 if the areas are exactly the same and 0 < similarness < 1.0 otherwise.

    """
    areas = individual_box_areas(boxes)

    area_max = np.maximum(areas[pair_idx[0]], areas[pair_idx[1]])
    area_min = np.minimum(areas[pair_idx[0]], areas[pair_idx[1]])
    pairwise_similarness = area_max / area_min

    return pairwise_similarness


def pairwise_box_edge_similarity_func(boxes, pair_idx):
    """
    calculate the pairwise minimum distance between edges of
    two individual boxes. This will NOT identify the distance of a box once they boxes touch
    each other. It only calculates the distance of the closest edges to each
    other.

    This function can be used to identify boxes where the "edges" are connected...

    a matrix will be calculated where each pair indicates the
    maximum of the two coordinates of the distance between the two closest edges.

    works pretty much like this:

           ┌─────────┐
           │     b3  │
           └▲────────┘
            │d
    ┌───────▼┐
    │  b1    │
    └────────┘      ┌─────────┐
             ◄──────►   b2    │
                d   └─────────┘

    As we are talking about boundingboxes with vertical/horizontal edges
    we can calculate the minimum distance of points more efficiently by
    getting the minimum distance for y and x coordinates individually
    and then applying a distance norm on the two resulting
    minimum distances in x- and y directions afterwards.
    in order to have a very "fast" distance norm we are using the max-norm
    but we could also use manhattan or euler instead...
    """

    x_min_diff = boxes[pair_idx[0], [x0, x1], None] - boxes[pair_idx[1], None, [x0, x1]]
    x_min_dist = np.abs(x_min_diff).reshape(-1, 4).min(1)
    y_min_diff = boxes[pair_idx[0], [y0, y1], None] - boxes[pair_idx[1], None, [y0, y1]]
    y_min_dist = np.abs(y_min_diff).reshape(-1, 4).min(1)

    # apply distance metric to y & x coordinates
    dist = np.maximum(x_min_dist, y_min_dist)
    return dist


def pairwise_is_completely_inside_matrix(boxes, eps: float = 10e-7):
    """pairwise check if boxes are completely inside of each other and return
    a matrix

    :param eps: add a very small tolerance to account for rounding errors):

    if a cell state is *True* it means, the box/index indicated by the row is
    contained/inside the box/index as indicated by the column

    checks if either b2 is inside b1 or b1 is inside b2:

     ┌─┬──────┬──────┐
     │ │  b2  │      │
     │ │      │  b1  │
     │ └──────┘      │
     └───────────────┘

    if one the two are completely inside the other one, the function returns "True"
    otherwise "False".

    if combined min bb area equals individual boundingbox in "areas",
    we can be sure that it contains the other box areas
    is applied *column-wise*.
    This also means that for each *row* it means the opposite:
    if the statement returns True, that means the box index described by a row
    is contained in another box. If the above statement is *true*.
    the following *is_inside* therefore states:
    "if a cell state is *True* it means, the box/index in the row is contained in
    the box/index in the column"
    """
    areas = individual_box_areas(boxes)
    dist = calc_pairwise_matrix(pairwise_minbb_areas, boxes, diag=areas)

    is_inside = (dist < areas + eps)
    return is_inside, (dist, areas)


def pairwise_box_overlap_distance(boxes: np.ndarray, pair_idx: np.ndarray):
    """
    calculate overlap ratio of boxe edges.

     ┌──────┐
     │   A1 │
     │  ┌───┼──────────┐ ▲     ▲
     │  │ OL│       A2 │ │ -dy │
     └──┼───┘          │ ▼     │h_min
        └──────────────┘       ▼
        ◄───►
         -dx
     ◄──────►
       w_min

    The ratio is defined by the minimum of the edges in each axis direction:

        r_x = -dx/min(e_x1,e_x2)
        r_y = -dy/min(e_y1,e_y2)

    The boxes only overlap if both distance are negative. This means
    we can simply take to get the actual overlap between them
    (if it is negative).

    d = max(r_x,r_y)

    if d is positive, it indicates the maximum distance between the boxes of either axis direction.
    """
    # TODO: maybe use the pairwise_box_alignement_along_axis for this?
    raise NotImplementedError("TODO")
    wh = individual_box_w_h(boxes)

    dx = pairwise_box_gap_distance_along_axis_func(boxes, pair_idx, axis=0)
    dy = pairwise_box_gap_distance_along_axis_func(boxes, pair_idx, axis=1)

    dx_min = np.min(boxes[pair_idx[0], [x0]])
    dy_min = np.min()

    return R


def pairwise_box_overlap_area_distance(boxes: np.ndarray, pair_idx: np.ndarray):
    """
    calculate overlap area ratio of boxes.

    ┌────────────────┐
    │      ┌─────────┼────┐ ▲
    │  A1  │   OL    │ A2 │ │ dy
    └──────┼─────────┘    │ ▼
           └──────────────┘
           ◄─────────►
               dx

    The ratio is defined by R = OL/(A1+A2).

    The problem with this function is that it requires "well-defined" boxes
    if we don't have boxes with a certain minimum area (for example lines
    as "infinitly thin" boxes) this function doesn't work anymore...

    If there is an overlap, the overlap area becomes negative.
    Otherwise it will be positive.
    """

    areas = individual_box_areas(boxes)
    dx = pairwise_box_gap_distance_along_axis_func(boxes, pair_idx, axis=0)
    dy = pairwise_box_gap_distance_along_axis_func(boxes, pair_idx, axis=1)

    # calculate pairwise overlap areas
    OL = dx * dy
    SIGN = 1 - 2 * ((dx < 0) & (dy < 0))  # only negative if both dx and dy are negative
    R = SIGN * OL / (areas[pair_idx[0]] + areas[pair_idx[1]])

    # box ONLY overlaps if BOTH distances are < 0
    # calculate overlap distance as the sum of dy & dy:
    return R


def pairwise_box_alignement_along_axis(boxes, pair_idx, axis, rel=False):
    """
    calculate the box alignement along *axis* as shown below (for y-axis):

    if axis=0, we are calculating alignement along x-axis
    (but individual distances value is in y-direction)

    ───────► x

            ▲  ┌───────┐
            │  │b3     │ dy > 0
        dy3 │  └───────┘
     ┌──────┴──┐
     │         ├──────────┐
     │   b1    │  b2      │  dy = 0
     │         ├──────────┘
     │         ├────────┐
     └─────┬───┤  b4    │
       dy4 │   │        │ dy > 0
           ▼   └────────┘

    where dy1_3 = bb_y_1_3 - max(e_y1, e_y3)

    with bb_y the bounding box length in axis direction and e_yXX the
    individual edge lengths of the boxes. In words:

    Subtract the edge of the shorter box from the boundingbox edge.

    The minimum of this function is 0 if the shorter box is fully contained
    inside the bounds of the longer box.
    """

    # length of minimum boundingbox along the axis
    a_al = 0 if axis == 1 else 1
    l_along_axis = pairwise_bbox_length_along_axis(boxes, pair_idx, a_al)
    lb = individual_box_length(boxes, a_al)

    max_edge = np.maximum(lb[pair_idx[0]], lb[pair_idx[1]])
    if rel:
        d = (l_along_axis - max_edge) / max_edge
    else:
        d = l_along_axis - max_edge

    return d


def pairwise_txtbox_dist(boxes, pair_idx, min_line_alignement, max_box_gap):
    """
    This function calculates the distance of two textboxes in order
    to figure out whether they belong in the same line or word or column...

        ┌───────┐
        │Box "A"│     ┌───────┐
        │       │     │Box "B"│
      ▲ └───────┘     │       │
      │min_line_alignement    │
      │               │       │
      ▼               └───────┘
                ◄─────►
              max_box_gap

    d = max(min_line_alignement, max_box_gap)

    """
    # calculate vertical character line overlap
    d_al = np.maximum(0, pairwise_box_alignement_along_axis(boxes, pair_idx, 0, rel=False))
    # calculate horizontal distance to next character
    d = np.maximum(0, pairwise_box_gap_distance_along_axis_func(boxes, pair_idx, 0))

    return np.maximum(
        d_al / min_line_alignement,
        d / max_box_gap
    )


def pairwise_edge_coordinate_alignement(boxes, pair_idx):
    """
    Calculates the pairwise alignement of the 4 edges and 2 middle lines
    of two boxes:

    vertical-left, vertical-middle, vertical-right,
    horizontal-bottom, horizontal-middle, horizontal-top
    """
    x_mean = (boxes[:, x0] + boxes[:, x1]) / 2.0
    y_mean = (boxes[:, y0] + boxes[:, y1]) / 2.0
    return np.array([
        np.abs(boxes[pair_idx[0], x0] - boxes[pair_idx[1], x0]),  # vertical-left
        np.abs(x_mean[pair_idx[0]] - x_mean[pair_idx[1]]),  # vertical-middel
        np.abs(boxes[pair_idx[0], x1] - boxes[pair_idx[1], x1]),  # vertical-right
        np.abs(boxes[pair_idx[0], y0] - boxes[pair_idx[1], y0]),  # horizontal-bottom
        np.abs(y_mean[pair_idx[0]] - y_mean[pair_idx[1]]),  # horizontal-middle
        np.abs(boxes[pair_idx[0], y1] - boxes[pair_idx[1], y1])  # horizontal-top
    ])


# use the query element as the first element of the data
# to be able to use the pairwise functions
def distance_query(
        q_elem: np.ndarray,
        data: np.ndarray,
        max_dist=None,
        k=None,
        dist_func=pairwise_euclidean_distance
):
    """
    This function leverages the above defined distance functions
    in order to execute small brute-force distance queries. If the number of elements
    being queried is small (<1000), this method is
    probably faster than using kd-trees or ann algorithms for
    (approximate) nearest neighbours.

    q_elem is
    """
    tmp_data = np.vstack((q_elem, data))
    pair_idx = linear_pairwise_indices(len(tmp_data))
    d = dist_func(tmp_data, pair_idx)
    # pair indices with distances
    res = np.vstack((d, pair_idx[1])).T
    # sort according to distance and remove the first element
    # which was the query element
    res = res[1:][res[1:, 0].argsort()]
    res[:, 1] -= 1  # subtract 1 from the indices to get the original indices of "data" back...
    if max_dist:
        res = res[res[:, 0] < max_dist]
    if k:
        return res[:k]
    else:
        return res


def distance_query_manhattan(
        q_elem: np.ndarray,
        data: np.ndarray,
        max_dist=None,
        k=None
):
    """Same as distance_query but with pre-defined manhattan distance function"""
    return distance_query(q_elem, data, max_dist, k, dist_func=pairwise_manhatten_distance)


distance_query_euclidean = distance_query


# TODO: replace all specialized functions below with this generic approach here
def calc_pairwise_matrix(pairwise_func: typing.Callable, data, diag=None, **kwargs):
    """
    Generic function which calculates pairwise matrices using a vectorized
    pairwise function and returns a matrix.

    Can be used for pairwise distance matrices for example...

    TODO: make a "preselection" possible where we first calculate a simple low-cost
          distance matrix, filter the pairs and based on that calculate the expensive solution for fewer
          pairs.
    """
    n = data.shape[0]

    # get lower triangle indices without diagonals
    # this makes our pair-wise calculation more efficient
    # and saves us more than half of our calculations
    tri_idx = lower_triangle_indices_cached(n)
    # calculate pairwise distance for triangle index
    d_tri = pairwise_func(data, tri_idx, **kwargs)
    # convert back to matrix with optional given diagonal
    dist = ltri2square(d_tri, n, tri_idx, diag=diag)

    return dist


def get_default_extraction_params():
    # TODO: get better default parameters by doing some optimization
    es = 50  # alignement sensitivity
    gs = 5.0  # gap sensitivity
    return {
        "va": [gs, es, es / 2, es],
        "ha": [gs, es, es / 2, es],
        "h_al": [gs, es],
        "v_al": [gs, es],
    }


def pairwise_weighted_distance_combination(
        boxes: np.ndarray, pair_idx: np.ndarray,
        parameter_list: typing.Dict[str, typing.List]  # function parameters
        # weights: np.ndarray,
        # func_list: typing.Callable,
        # axis=0,
) -> np.ndarray:
    """
    combines multiple weighted distance metrics.
    This makes this function adaptable using optimizers.

    this one also works for lines if they are defined as boxes width width/height=0...

    TODO: write a "test-function" as a decorator which checks how many parameters we need for this function
          by running it once with a list which tracks the maximum index that was called from it...

    """
    boxes = boxes.copy()
    x_gap_dist = np.maximum(0, pairwise_box_gap_distance_along_axis_func(boxes, pair_idx, axis=0))
    y_gap_dist = np.maximum(0, pairwise_box_gap_distance_along_axis_func(boxes, pair_idx, axis=1))
    edge_alignements = pairwise_edge_coordinate_alignement(boxes, pair_idx)
    d_coll = []  # define a distance collection
    # TODO: make more efficient (for example we can probably add the x_gap/y_gap) at the end....
    #       as it is always the same and thus not subject to the min function...
    if p := parameter_list.get('va', False):
        # check vertical alignement
        d_coll += [p[0] * y_gap_dist + np.min(edge_alignements[:3] * [[p[1]], [p[2]], [p[3]]], axis=0)]
    if p := parameter_list.get('ha', False):
        # check horizontal alignement
        d_coll += [p[0] * x_gap_dist + np.min(edge_alignements[3:] * [[p[1]], [p[2]], [p[3]]], axis=0)]
    if p := parameter_list.get('h_al', False):
        # check distance/alignement/overlaps to neighbour
        d_coll += [p[0] * x_gap_dist + p[1] * pairwise_box_alignement_along_axis(axis=1)]
    if p := parameter_list.get('v_al', False):
        d_coll += [p[1] * y_gap_dist + p[2] * pairwise_box_alignement_along_axis(axis=0)]

    # TODO: add option, whether we would like to use "min" or "sum" here....
    # return method(np.dot(d, weights))
    # we are using the "minimum" of all the required distance to resemble an *or* relationship
    # for clustering: "put box in group if distance1 is small OR distance2 OR distance3...
    return np.min(d_coll, axis=0)


# TODO: add chain-combination function similar to how we have "layers" in
#       a neural network
#       with this method we would chain several layers of distance combinations
#       our scenario would be:
#       first calculate several different distance metric *sums* and then chain
#       those into a "minimum" function.


def boundarybox_query(bbs, bbox, tol=10.0, exclude=False):
    """
    This function filters a pandas list of boundingboxes for
    boxes that are fully contained in a specific region

    TODO: also do this using an rtree/bvh function

    :param bbs: list of boundary boxes to search
    :param bbox: search in this area
    :param tol: search tolerance (with respect to a single dimension)
    :return: list of boundary boxes extracted from bbs which are in the search area *bbox*
    """
    # valid_areas.loc[valid_areas.x0>bbox[0]].loc[valid_areas.x1<bbox[2]]
    # in order to increase the speed we filter with several .loc operations
    if bbs.empty:
        return pd.DataFrame()
    if exclude:
        return bbs.loc[~((bbs.y0 > (bbox[1] - tol)) & (bbs.y1 < (bbox[3] + tol))
                         & (bbs.x0 > (bbox[0] - tol)) & (bbs.x1 < (bbox[2] + tol)))]
    else:
        return bbs.loc[(bbs.y0 > (bbox[1] - tol)) & (bbs.y1 < (bbox[3] + tol))
                       & (bbs.x0 > (bbox[0] - tol)) & (bbs.x1 < (bbox[2] + tol))]


def boundarybox_intersection_query(bbs, bbox, tol=1.0):
    """
    This function filters a pandas list of boundingboxes for
    boxes that intersect with each other

    :param bbs: list of boundary boxes to search
    :param bbox: search in this area
    :param tol: search tolerance (with respect to a single dimension)
    :return: list of boundary boxes extracted from bbs which are in the search area *bbox*
    """
    # valid_areas.loc[valid_areas.x0>bbox[0]].loc[valid_areas.x1<bbox[2]]
    # in order to increase the speed we filter with several .loc operations
    indices = bbs.loc[bbs.y1 > (bbox[1] - tol)].loc[bbs.y0 < (bbox[3] + tol)] \
        .loc[bbs.x1 > (bbox[0] - tol)].loc[bbs.x0 < (bbox[2] + tol)].index

    return indices


# TODO: check in different location whether it makes sense that
#       we use a custom distance function in order to improve clustering
# TODO: directly hand over the distance matrix for clarity?...
def distance_cluster(data: np.ndarray = None,
                     distance_threshold: float = 0.0,
                     distance_func=None,
                     pairwise_distance_func=None,
                     distance_matrix=None):
    """
    This function takes a distance function specified by either:
    "distance_func" or "pairwise_distance_func" and uses them in oder to
    calculate a distance matrix which then gets used for an Agglomerative clustering
    algorithm.

    TODO: replace all occurences of clustering with this function
    clusters "box"-data according to a certain distance function

    TOOD: get rid of "AgglomerativeClustering by a custom function...

    TODO: only allow our vectorized_distance_functions
    """
    if pairwise_distance_func:
        # we can set the diagonal of the pairwise distance matrix to 0
        # as that distance *should* always be zero
        distance_matrix = calc_pairwise_matrix(pairwise_distance_func, data, diag=0)
    elif isinstance(distance_func, str):
        distance_matrix = pairwise_distances(data, metric=distance_func)

    if len(distance_matrix) > 1:
        # TODO: replace with our own clustering function
        #       as we already have the distance matrix, this shouldn't be too difficult...
        res = cluster.AgglomerativeClustering(
            n_clusters=None,
            linkage="single",  # {‘ward’, ‘complete’, ‘average’, ‘single’}, default=’ward’
            # we are choosing "manhattan" distance as we want to put emphasis on vertical/horizonal lines
            metric="precomputed",  # “euclidean”, “l1”, “l2”, “manhattan”, “cosine”, or “precomputed”
            connectivity=None,  # to make calculations faster
            compute_full_tree=True,  # has to be true when used together with distance_threshold
            distance_threshold=distance_threshold
        ).fit(distance_matrix)
    else:
        return [0], [0.]

    return res.labels_, res.distances_


# TODO: generalize this method to more dimensions in order to be able
#       to replace sklearn clustering for our distance functions...
# TODO: move this to list functions....
def cluster1D(data: np.array, agg_func: typing.Callable, merge_tol=5.0) -> np.array:
    x = np.unique(data, axis=0)
    # TODO: make distance calculation replaceable
    d = np.diff(x, axis=0).flatten()
    splits = np.argwhere(d > merge_tol).flatten() + 1
    bb_groups = np.split(x, splits.flatten())
    return np.array([agg_func(bb) for bb in bb_groups])


def merge_groups(df: pd.DataFrame, group_labels: str, **kwargs) -> Tuple[List[np.ndarray], np.ndarray]:
    # need to sort into groups in order for the numpy function to be able
    # to group labels
    df = df.dropna(subset=[group_labels]).sort_values(group_labels)
    if df.empty:
        return []  # return empty dataframe
    group_split_indices, group_sizes = np.unique(
        df[group_labels].values,
        return_index=True, return_counts=True)[1:]
    # TODO: also allow for dataframes to be split here...
    # groups = np.split(df[box_cols].values, group_split_indices[1:])
    groups = np.split(df.drop(columns=group_labels).values, group_split_indices[1:])
    return groups, group_sizes


def merge_bbox_groups(df: pd.DataFrame, group_labels: str, **kwargs) -> pd.DataFrame:
    """
    Merges grouped boxes defined by x0,y0,x1,y1 and group labels
    and returns the groups boundingbox
    """
    # ##  merge bounding boxes with same labels into large bounding box
    # originally this was done using the pandas groupby function inside
    # merge_bbox_groups_slow, but
    # this method was too slow for our purposes.

    bb_groups, group_sizes = merge_groups(df[box_cols + [group_labels]], group_labels)
    # TODO: implement optional metrics for each group (such as variance which is
    #       roughly equal to the accuracy of the group)
    merged_bboxes = pd.DataFrame([
        np.hstack((
            g[:, :2].min(0),
            g[:, 2:].max(0),
        )) for g in bb_groups],
        columns=["x0", "y0", "x1", "y1"])
    merged_bboxes['num'] = group_sizes
    return merged_bboxes
