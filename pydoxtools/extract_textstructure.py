import typing

import pandas as pd
import pdfminer


def _filter_boxes(
        boxes: pd.DataFrame,
        min_aspect_ratio=None,
        max_aspect_ratio=None,
        min_num=None,
        min_area=None,
):
    """filter boxes for various criteria

    TODO: optionally include more filter criteria...
        - min size
        - min margin
        - min size
        - max size
        - min_width
        - min_height
    """
    if not boxes.empty:
        if min_num:
            boxes = boxes[boxes["num"] >= min_num].copy()
        boxes['w'] = boxes.x1 - boxes.x0
        boxes['h'] = boxes.y1 - boxes.y0
        boxes['aspect_ratio'] = boxes.h / boxes.w
        boxes['area'] = boxes.w * boxes.h
        if min_area:
            boxes = boxes.loc[boxes.area > min_area]
        if min_aspect_ratio:
            boxes = boxes.loc[boxes.aspect_ratio > min_aspect_ratio]
        if max_aspect_ratio:
            boxes = boxes.loc[boxes.aspect_ratio < max_aspect_ratio]
    else:
        boxes = pd.DataFrame()

    return boxes


def _generate_text_boxes(df: pd.DataFrame) -> pd.DataFrame:
    """
    create textboxes and create bounding boxes and aggregated text from
    a pandas dataframe with textlines.
    returns a list of textboxes together wth some coordinate data and
    the contained text.

    TODO: integrate this function with the classes above..

    TODO: make it possible to use alternative distance metrics to generate
          the text boxes...
    """
    if "boxnum" in df:
        group = df.groupby(['p_id', 'boxnum'])
        # aggregate object from the same box and calculate new
        # bounding boxes, also join the formatted text
        bg = group.agg(
            x0=("x0", "min"), y0=("y0", "min"),
            x1=("x1", "max"), y1=("y1", "max"),
            text=("lineobj",
                  lambda x: "".join(_line2txt(obj) for obj in x.values))
        )
        # remove empty box_groups
        bg = bg[bg.text.str.strip().str.len() > 1].copy()
        # do some calculations
        bg['y_mean'] = bg[['y0', 'y1']].mean(axis=1)
        bg['x_mean'] = bg[['x0', 'x1']].mean(axis=1)
        bg['w'] = bg.x1 - bg.x0
        bg['h'] = bg.y1 - bg.y0
        return bg
    else:
        return pd.DataFrame(columns=['text'])


def _line2txt(LTOBJ: typing.Iterable):
    """
    extract text from pdfiner.six lineobj including size hints

    TODO: speedup using cython/nuitka/numba
    """
    txt = ""
    last_size = 0
    for i, ch in enumerate(LTOBJ):
        newtxt = ""
        sizehint = ""
        if isinstance(ch, pdfminer.layout.LTText):
            newtxt = ch.get_text()
        if isinstance(ch, pdfminer.layout.LTChar):
            newsize = ch.size
            if i > 0:
                # TODO: use an iterative function here...
                if newsize < last_size:
                    sizehint = "<s>"
                elif newsize > last_size:
                    sizehint = "</s>"
            last_size = newsize
        txt += sizehint + newtxt
    return txt
