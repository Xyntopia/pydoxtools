from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import functools
import hashlib
import logging
import re
import typing
from functools import cached_property

import PIL.Image
import numpy as np
import pandas as pd
import pydantic
from pdfminer.layout import LTChar
from sklearn.neighbors import KernelDensity

import pydoxtools
import pydoxtools.document_base
import pydoxtools.operators_base
from pydoxtools import cluster_utils as gu
from pydoxtools.cluster_utils import pairwise_txtbox_dist, box_cols, y1, x0, x1, boundarybox_intersection_query
from pydoxtools.extract_html import extract_lists, extract_tables
from pydoxtools.extract_textstructure import _line2txt
from pydoxtools.operators_base import Operator

logger = logging.getLogger(__name__)

idx = pd.IndexSlice


class TableExtractionParameters(pydantic.BaseModel):
    # possible values: "t", "g", "tg", with "t" for text-based and "g" for graphics based
    # TODO: enable t and b separatly for table area detection and cell extraction
    extraction_method: str = "g"
    min_table_area: float = 50.0 * 50.0  # in dots * dots, using pdf coordinates (72dpi)
    # aspect ratios are calculated as h/w
    min_aspect_ratio: float = 1.0 / 50.0  # exclude for example single lines
    max_aspect_ratio: float = 50.0  # exclude for example high graphic elements...
    # parameters for the distance function
    # a list of dicts parameters for each iteration where each dict includes
    # every used sub-distance-function, where each key in the dictionary
    # resembles the function and the values the parameters for that specific function
    # TODO: document this in a better way...
    area_detection_distance_func_params: typing.List[typing.Dict[str, typing.List[float]]]
    # on top of the table area we put a margin which helps
    # when area detection was inaccurate
    text_extraction_margin: float = 10.0

    #  #table cell extraction
    max_v_line_thickness: float = 5.0  # maximum thickness which is allowed for a vertical line
    elem_scan_tol: float = 1.0  # used in several places where we want to know coordinates of lines

    # the x-cursor has to advance at least this far in order for anew cell to be created
    # otherwise it simply omits the vertical line element.
    min_cell_width: float = 6.0

    # tolerance which determines which text lines will be added to a cell area
    # this should stay below a typical line-height in pixels (>8 )in order to
    # avoid adding neighbouring textlines...
    text_line_tol: float = 5.0

    # table conversion to dataframe
    table_line_merge_tol: float = 7.0  # to which extend will neighbouring cell lines be merged into a table-column/row?
    cell_idx_tol: float = 5.0  # how precise should the coordinates be for each cell in relation to the table lines?

    # TODO: make these parameters relative
    # min_char_line_overlap the tolerance to which extend cahracters have to be in the same
    # line to be considered part of the same word
    max_char_disalignement: float = 4.0
    # maximum distance of characters to be considered in the same line...
    # this should be low for tables...
    max_char_dist: float = 2.0

    @classmethod
    def reduced_params(cls):
        hp = {'es1': 11.1, 'es2': 2.1, 'gs1': 11.1, 'gs2': 20.1}
        adp = [{
            "va": [hp['gs1'], hp['es1'], hp['es1'] / 2, hp['es1']],
            "ha": [hp['gs1'], hp['es1'], hp['es1'] / 2, hp['es1']]
        },
            {
                "va": [hp['gs2'], hp['es2'], hp['es2'] / 2, hp['es2']],
                "ha": [hp['gs2'], hp['es2'], hp['es2'] / 2, hp['es2']]
            }]
        return cls(
            area_detection_distance_func_params=adp,
        )


def _calculate_ge_points_density_metrics(points):
    """calculate vertical point density (edge density would be without vertical lines)

    a,b = _calculate_ge_points_density_metrics(points[["y0","y1"]].values.flatten())
    pd.DataFrame([a,b]).T.plot(x=0,y=1)

    # TODO: use point density for caculations (for example which tables are valid?)
    """
    # get unique point with their count and store it in an array
    ps = np.array(np.unique(points, return_counts=True))
    # the following calculates a density function with a bandwidth of 0:
    # we replaced this with the KDE below to get better results.
    # pd = ps[1]/np.diff(y[0], append=[0])

    # because we use "tophat" function, badwidth can be interpreted as pdf-pixels
    # (standard pdf coordinate system at 72dpi)
    bw = 2.0
    kde = KernelDensity(bandwidth=bw, kernel="tophat", rtol=0.5, metric="manhattan").fit(ps[0][:, None],
                                                                                         sample_weight=ps[1])
    ge_density = np.exp(kde.score_samples(ps[0][:, None])) * bw
    # pd.DataFrame(y).T.plot(x=0,y=1)
    # pd.DataFrame([y[0],gaussian_kde(y[0], 0.1, y[1])(y[0])]).T.plot(x=0,y=1)
    return ps[0], ge_density


def _get_cell_text(cell, min_new_line_gap=4, strip=False):
    # TODO: make sure we preserve textblocks...
    cell["sortkey"] = cell.x0 - 40 * cell.y0
    sorted_cell = cell.sort_values(by="sortkey").reset_index(drop=True)
    # check for newlines
    if strip:
        sorted_cell.text = sorted_cell.text.str.strip()
    newline = sorted_cell.y0.diff() < -min_new_line_gap
    cell_text = "".join(("\n" if nl else " ") + t for t, nl in zip(sorted_cell.text, newline))[1:]
    return cell_text


def _get_cell_coordinates(cell_edges, table_lines, tol=5.0):
    c = cell_edges + [+tol, -tol]  # make cells "thinner" to more easily fit inside table_lines
    c0, c1 = c.T
    tc = np.argwhere((c0[:, None] < table_lines[1:]) & (c1[:, None] > table_lines[:-1]))
    tc = np.split(tc[:, 1], np.unique(tc[:, 0], return_index=True)[1][1:])
    return tc


def _LTObj2Chars(df_le: pd.DataFrame) -> pd.DataFrame:
    chars = []
    for text_line in df_le.obj.values:
        for character in text_line:
            if isinstance(character, LTChar):
                chars.append(character)
    chars = pd.DataFrame([dict(
        obj=c,
        x0=c.bbox[0],
        y0=c.bbox[1],
        x1=c.bbox[2],
        y1=c.bbox[3],
    ) for c in chars])
    return chars


def get_horizontal_row_elements(
        ge: pd.DataFrame, y0_cursor: float,
        max_v_line_thickness: float, elem_scan_tol: float, min_cell_width: float):
    # left_over_y1 are the graphics elements that we can find "above" the y-cursor
    # we start by filtering for y1 which is all elements that "end" above
    # our y0-cursor
    left_over_y1 = ge.loc[idx[:, y0_cursor:], :]
    if left_over_y1.empty:
        # assume we are in the last box and need to
        # close up any remaining boxes...
        return None
    # then filter for all elements that "start/y0" after our y0_cursor
    left_over_y0 = left_over_y1[y0_cursor:]
    next_h_elem = left_over_y1.iloc[0]
    # TODO: do we need a min here?
    # get the y-coordinate of closest horizontal element in case "empty y0", we get the y1-coordinate
    # e.g. if there is a box at the top of the table and there are no ore y0-coords left-over
    y0_h_elem = next_h_elem.y1 if left_over_y0.empty else min(next_h_elem.y1, left_over_y0.iloc[0].y0)
    # TODO: we might want to check if *max_v_line_thickness* should be the
    #       distance to the next text element.... but maybe its enough to simply
    #       have this as a parameter...
    # select horizontal elements in row
    h_row_elem = left_over_y1.loc[
                 idx[: y0_h_elem + max_v_line_thickness,
                 y0_h_elem - elem_scan_tol:],
                 :]
    # sort our h_row_elem with small cell widths
    h_row_elem = h_row_elem.loc[
        h_row_elem.w > min_cell_width]  # .reorder_levels([2, 3, 0, 1], axis=0).sort_index()
    # and extract all relevant horizontal lines from them for this row that can be used to close cells
    h_lines = pd.concat([
        # select for y0
        h_row_elem.loc[y0_h_elem - elem_scan_tol:y0_h_elem + max_v_line_thickness,
        ["x0", "x1", "y0"]].rename(columns={"y0": "y"}),
        # select for y1
        h_row_elem.loc[idx[:, y0_h_elem - elem_scan_tol:y0_h_elem + max_v_line_thickness],
        ["x0", "x1", "y1"]].rename(columns={"y1": "y"})
    ]).droplevel(["y0", "y1"]).sort_index().drop_duplicates(["x0", "x1"])

    return y0_h_elem, h_lines


def _close_cell(oc, df_le, y1, text_line_tol):
    # the current indexing of text lines is the following:
    # "y0","x0","y0","x1" and so we need to
    # to do the indexing accordingly
    # TODO: not sure if "min" is the right operation here, it might also be enough
    #       to just use "iloc[0]" due to already correct sorting...
    oc['y1'] = y1
    text_elements = df_le.loc[idx[
                              oc["y0"] - text_line_tol:,
                              oc["x0"] - text_line_tol:,
                              :oc["y1"] + text_line_tol,
                              :oc["x1"] + text_line_tol],
                    :]
    if text_elements.empty:
        return None
    else:
        oc['text_elements'] = text_elements
        return oc


def _close_open_cells(open_cells, h_lines, df_le, elem_scan_tol,
                      text_line_tol, y0_cursor):
    # iterate through each open cell and check which ones we can close with the
    # provided top_cell_border_elements
    still_open = []
    new_cells = []
    for oc in open_cells:
        # check if we have a top element (h_lines) that somehow exists within the cell borders
        # and can be used to close the cell...
        # we have 3 cases:
        #  - both ends are inside the cell
        #  - x0 is on the left side of the left line end
        #  - x1 is on the right side of the right line end
        # all three cases are handled by the expression below:
        top_elem = h_lines.loc[
                   idx[:oc['x1'] - elem_scan_tol,
                   oc['x0'] + elem_scan_tol:],
                   :]

        if top_elem.empty:
            still_open.append(oc)
        else:  # if top_elem exits, close the cell
            if new_cell := _close_cell(oc, df_le, y0_cursor, text_line_tol):
                new_cells.append(new_cell)

    return new_cells, still_open


class ListExtractor(Operator):
    """
    Extract lines that might be part of a "list".
    """

    def __call__(self, line_elements: list[pydoxtools.document_base.DocumentElement]) -> pd.DataFrame:
        # search for lines that are part of lists
        # play around with this expression here: https://regex101.com/r/xrnKlm/1
        degree_search = r"^[\-\*∙•](?![\d\-]+\s?(?:(?:[°˚][CKF]?)|[℃℉]))"
        df_le = pd.DataFrame(line_elements)
        has_list_char = df_le.rawtext.str.strip().str.contains(degree_search, regex=True, flags=re.UNICODE)
        list_lines = df_le[has_list_char].rawtext.str.strip().to_frame()

        return list_lines


class HTMLTableExtractor():
    def __call__(self):
        # TODO: put this into the html pipeline...

        lists = extract_lists(raw_html)
        html_tables = []
        for html_con in [main_content_html, main_content_html2]:
            html_tables.extend(extract_tables(html_con))

        # add lists to tables
        tables = lists + html_tables
        # convert tables into json readable format
        tables = pydoxtools.list_utils.deep_str_convert(tables)


class PDFTableCandidate:
    def __init__(
            self, df_le, df_ge, initial_area: np.ndarray,
            tbe: TableExtractionParameters = None,
            page: int = None, page_bbox=None, file_name=None
    ):
        """
        Convert an area list of graphic and line elements into a table.

        page, page_bbox, filename are all used for debug-purposes

        TODO: make text-only cell extraction possible
        TODO: detect text boxes from lines...
        TODO: detect standard distance between table lines in order to
              detect rows and use that for better span cell detection
              (which presumably has smaller distances between textlines)
              after we have the "standard distance" we would merge
              lines that are below this limit into a textbox...
        TODO: detect alignement of cells and use that for row/colum detection
              e.g. we could detect aligned cells (left/center/right)
              for columns. then we build a table of all textblocks and
              which alignement they belong to...
              Then we start with a cursor text-block-by-text-block scanning towards the
              right and adding them into their respective rows...
        TODO: we could do a graphical cell-detection & a text-only detection and
              then overlay the on top of each other. Text-only detections
              could be merged with the cell detections. where cell-detections
              should make the text-only-detections more precise...
              for example we could check for each text-only-cell wether it is part
              of a larger graphics-only cell. Or we could check if a graphics cell
              splits a text-cell in half or something like that...
        """
        self._filename = file_name
        self._page_bbox = page_bbox
        self._df_le = df_le
        self._page = page
        self._df_ge = df_ge
        self._tbe = tbe or TableExtractionParameters.reduced_params()
        self._initial_area = initial_area

        # TODO: put into table extraction parameters
        self.max_lines = 1000

        self._debug = {}

    @property
    def tbe(self) -> TableExtractionParameters:
        return self._tbe

    @cached_property
    def df_le(self) -> pd.DataFrame:
        """line elements of table"""
        return gu.boundarybox_query(
            self._df_le, self._initial_area,
            tol=self.tbe.text_extraction_margin
        ).copy()

    @property
    def page(self):
        return self._page

    @cached_property
    def df_ge(self) -> pd.DataFrame:
        """graphic elements of table"""
        # TODO: maybe we should not use filtered, but unfiltered graphic elements here?
        return gu.boundarybox_query(
            self._df_ge, self._initial_area,
            tol=self.tbe.text_extraction_margin
        ).copy()

    @cached_property
    def df_ch(self) -> pd.DataFrame:
        """access to individual characters of the table. We can use this
        to apply our own layout algorithm, because the standard pdfminer.six
        algorithms don't work as well here...
        """
        return _LTObj2Chars(self.df_le)

    @cached_property
    def df_words(self):
        """
        Calculate word boxes instead of line boxes for tables
        this is important here, as we would like textboxes to be split according to detected
        cells. the standard algorithms of pdfminer.six don't do this
        as well...
        """

        char_dist_matrix = gu.calc_pairwise_matrix(
            pairwise_txtbox_dist,
            self.df_ch[box_cols].values,
            diag=0,
            min_line_alignement=self.tbe.max_char_disalignement,
            max_box_gap=self.tbe.max_char_dist
        )

        max_word_distance = 1.0
        self.df_ch["groups"], dist_m = gu.distance_cluster(
            distance_matrix=char_dist_matrix, distance_threshold=max_word_distance
        )
        # create new column with the type of group (hb,hm,ht,vb,vm,vt) and their labels
        bb_groups, group_sizes = gu.merge_groups(self.df_ch, "groups")

        # char_list = [[] for c in bb_groups]
        word_boxes = pd.DataFrame([
            (  # calculate new boundingbox
                g[:, 1].min(0),
                g[:, 2].min(0),
                g[:, 3].max(0),
                g[:, 4].min(0),
                g[g[:, 1].argsort()][:, 0]  # sort char boxes in the right order
            ) for g in bb_groups],
            columns=["x0", "y0", "x1", "y1", "chars"])
        word_boxes["text"] = word_boxes.chars.apply(lambda x: _line2txt(x).strip())

        return word_boxes

    @cached_property
    def bbox(self) -> np.ndarray:
        # find the exact bounding box of our table...
        dims = []
        dims += [[self.df_le.x0.min(), self.df_le.y0.min(), self.df_le.x1.max(), self.df_le.y1.max()]]
        if not self.df_ge.empty:
            dims += [[self.df_ge.x0.min(), self.df_ge.y0.min(), self.df_ge.x1.max(), self.df_ge.y1.max()]]
            dims = np.array(dims)
            return np.hstack((np.nanmin(dims[:, :2], 0), np.nanmax(dims[:, 2:], 0)))
        else:
            return np.array(dims[0])

    @functools.lru_cache
    def detect_cells(self, steps=None) -> pd.DataFrame:
        """
        This algorithm works by slowly scanning through a table bottom-to-top first
        and left-to-right for each row.

        On the way we detect rows by alternating between text lines and graphical lines.
        We can detect a cell border by going from the bottom of a text-line-box to the bottom
        of the next graphical element box in that specific cell in both directions.
        All other textlines that are in between there should be part of that same cell.

        Additionally, we can efficiently assign textlines to cell areas while iterating
        through the rows.

        Once we have identified all cells and their borders
        we can infer their cell indices. By clustering their cell borders and identifying
        cells indices afterwards.

        TODO: add an ascii sketch about the process...

        TODO: we need to do something in the case where we have a table with actual
              box elements as table-rows/borders. in that case we can not use x0/x1 anymore..
        """

        max_v_line_thickness = self.tbe.max_v_line_thickness  # maximum thickness which is allowed for a vertical line
        elem_scan_tol = self.tbe.elem_scan_tol
        min_cell_width = self.tbe.min_cell_width
        text_line_tol = self.tbe.text_line_tol  # tolerance which detremines which text lines will be added to a cell area

        # TODO: use the "area-ratio" of elments in order to improve quality?
        # create indices for our graphical and line elements for
        # faster spatial queries
        # TODO: we might not need "w" and "h" or other elements...
        # self.df_ge[['w', 'h']] = self.df_ge[['x1', 'y1']] - self.df_ge[['x0', 'y0']]
        if self.df_ge.empty:
            return pd.DataFrame()
        ge = self.df_ge[box_cols + ['w', 'h']].sort_values(
            by=['y0', 'y1', 'x0', 'x1'],
        ).set_index(['y0', 'y1', 'x0', 'x1'], drop=False).copy()
        le = self.df_words[box_cols + ['text']].sort_values(
            by=["y0", "x0", "y1", "x1"],
        ).set_index(['y0', 'x0', 'y1', 'x1'], drop=False).copy()

        # generate a list of increasing y-mean values for all lines which we can "pop"
        # from the list in order to scan the table upwards...
        y_mean = (le.y1 + le.y0) / 2.0
        y_mean_list = y_mean.drop_duplicates().sort_values(ascending=False).to_list()

        open_cells = []  # cells that the cursor is currently traversing
        # finished cells, where the y0-scan cursor has passed
        # by the top y1 border and "closed" them this way...
        cells = []

        # move through the table bottom-to-top evaluating row-by-row
        # define a y- and x-cursor which we will slowly advance in order to scan the table
        # we start with the first text element in the lower left
        y0_h_elem = 0.0

        max_steps = steps or self.max_lines  # maximum of 1000 lines for a table for safety...
        for i in range(max_steps):
            if not y_mean_list:  # if y_mean_list is empty, we reached the top of the table
                # as there are no more text lines we can close up all cells
                # that are still open
                last_cells = [_close_cell(oc, le, self.bbox[y1], text_line_tol) for oc in open_cells]
                cells.extend(lc for lc in last_cells if lc)
                break

            # advance y0_cursor to the next textline
            # we stop as soon as our cursor points to a higher location than
            # the top of our last textline y0_h_elem
            # TODO: use y_mean to the the left_over_lines!!
            while y_mean_list:  # do this until we don't have any more text elements left
                y0_cursor = y_mean_list.pop()
                # make sure that the middle of the line is above the previously found table row
                if y0_cursor > y0_h_elem:
                    break

            # now we would like to know every vertical line (not element) that crosses the current
            # y0-cursor-line to get the vertical cell borders in this row...
            # this works because we are using the center of cells...
            # get all elements reaching from `below` to `above + tolerance` (using y0 and y1)
            # we do "swaplevel" in order to be able to use more efficient x-indexing afterwards
            row_ge_elem = ge.loc[idx[:y0_cursor + elem_scan_tol, y0_cursor + elem_scan_tol:], :]
            v_row_elem = row_ge_elem.swaplevel(0, 2, axis=0).sort_index()

            # TODO: we should be doing something different here when not detecting any graphical elements!
            # TODO: also think about table were we the lines are only partially indicated by graphics elements
            #       so..  we should accept cell borders to be "either" text "or" graphics based...
            if v_row_elem.empty:
                # if no vertical line was detected in this row, so we should move further upwards
                # and set y0_h_elem to the top of this line element.
                if y_mean_list:
                    y0_h_elem = (y0_cursor + y_mean_list[-1]) / 2
                continue

            # get all vertical coordinates because we would like to identify the lines and not
            # boxes...
            # TODO: we should probably also detect text boxes here with their vertical lines!
            vlines = np.sort(np.hstack(
                (np.unique(v_row_elem[['x0', 'x1']].values), self.bbox[[x0, x1]])))
            x0_cursor = vlines[0]
            # now scan the table row horizontally and
            # add "open" cells for every vertical line that we detect, except the ones we "already have"
            for x in vlines:
                # TODO: take txt_lines  here into account. For example:
                #       if a cell has a textline which is longer than its x1-border,
                #       we should make the cell longer...
                if x < x0_cursor + min_cell_width:
                    x0_cursor = x
                    continue

                # check if this cell already exists in "open cells"
                # by checking if our x-cursor is inside that cell
                # if not, create it..
                if not next((oc for oc in open_cells
                             if oc['x0'] <= (x0_cursor + elem_scan_tol) <= oc['x1']), None):
                    # set cell borders but leave cells open as we have to advance our y0 cursor
                    # to find out the y1 value of the cells ("close" them)...
                    cell = dict(
                        x0=x0_cursor,
                        y0=y0_h_elem,
                        x1=x  # use left side of broder element as right side of cell
                    )
                    open_cells.append(cell)
                x0_cursor = x  # use right side as the next x0_cursor for left-side of the next cell

            # now after scanning the row and creating "open cells", collect
            # all horizontal elements above our current cursor...
            # advance y-cursor upwards
            # and check which cells we can close... # get the next horizontal element
            # TODO: handle the case where we have text "above" graphic lines...
            # TODO: might need to check for y1 here as well, as there
            #       might be a case were the box only closes and no new
            #       one opens... more tests will show...
            #       maybe check whatever is between here and the next text element?
            #       and then take the minimum y-coordinate from that..
            #       YEAH --> we need this.. already our first test showed :P
            if res := get_horizontal_row_elements(ge, y0_cursor,
                                                  max_v_line_thickness, elem_scan_tol, min_cell_width):
                y0_h_elem, h_lines = res
            else:
                continue

            new_cells, still_open = _close_open_cells(
                open_cells, h_lines, le, elem_scan_tol, text_line_tol, y0_h_elem)
            open_cells = still_open
            cells.extend(new_cells)

        if steps:  # return additional debug information
            self._debug["open_cells"] = pd.DataFrame(open_cells)
            self._debug["open_cells"]['y1'] = self.bbox[y1]
            return pd.DataFrame(cells)
        else:
            return pd.DataFrame(cells)

    @functools.lru_cache
    def convert_cells_to_df(self) -> typing.Tuple[pd.DataFrame, typing.Tuple]:
        """
        TODO: make the algorithm work with non-graphical tables as well by searching for rows/columns/cells
              using textboxes only.

        TODO: enhance function by also taking textboxes into account. (for large table cells)
        """
        """TODO: is the "loop"-approahc faster?


        #table = np.empty((len(hlines)-1,len(vlines)-1), dtype=object)
        #x_cells = (cell.x0<vlines[1:]) & (cell.x1>vlines[:-1])
        #y_cells = (cell.y0<hlines[1:]) & (cell.y1>hlines[:-1])[::-1] # reversed because we want table coordinates to be from top to bottom
        #table[y_cells,x_cells] = cell_text
        """

        cells = self.detect_cells().copy()
        if cells.empty or len(cells) < 2:
            return pd.DataFrame(), ([], [], cells)
        cells['text'] = cells["text_elements"].apply(_get_cell_text)

        # after having detected all cells, we need to order them
        # into a table
        # first thing we need to do is to cluster cell borders
        x_coordinates = cells[["x0", "x1"]].values
        y_coordinates = cells[["y0", "y1"]].values

        # now create the clusters in order to detect horizonal & vertical lines in
        # the table
        vlines = gu.cluster1D(x_coordinates.reshape(-1, 1), np.mean, self.tbe.table_line_merge_tol)
        hlines = gu.cluster1D(y_coordinates.reshape(-1, 1), np.mean, self.tbe.table_line_merge_tol)

        # generate cell coordinates
        xtcs = _get_cell_coordinates(x_coordinates, vlines, tol=self.tbe.cell_idx_tol)
        ytcs = _get_cell_coordinates(y_coordinates, hlines, tol=self.tbe.cell_idx_tol)

        # --> sometimes we create "weired" vertical/horzontal lines by averaging them.
        # in that case some cells don't fit into the raster anymore and we have too few lines for every
        # cell. usually this happens in areas that are not real tables such as figures...
        # so this is a rally nice way to sort som of them out ;).
        # TODO: maybe at some point in the future we need a more robust method to create
        #       coordinate grids though ...
        if (len(xtcs) != len(ytcs)) or (len(xtcs) < len(cells)) or (len(ytcs) < len(cells)):
            return pd.DataFrame(), ([], [], cells)

        cells['coords'] = [np.stack(np.meshgrid(xtc, ytc), -1).reshape(-1, 2) for xtc, ytc in zip(xtcs, ytcs)]
        t = cells[['coords', 'text']].explode("coords")
        t[['x', 'y']] = t.coords.apply(pd.Series)
        table = pd.pivot_table(t, values='text',
                               index=['y'], columns=['x'],
                               aggfunc='first', fill_value='')
        table = table.sort_values('y', ascending=False).reset_index(drop=True)
        return table, (hlines, vlines, cells)

    @functools.lru_cache
    def convert_cells_to_df_text_only(self) -> typing.Tuple[pd.DataFrame, typing.Tuple]:
        """
        extract tables using only textboxes

        TODO: make sure we can also detect span-cells and other more difficult-to-detect stuff
        """
        # detect vertical lines for column detection
        # we are using lines here, but it might be better in the future to use
        # words for this  if some lines get detected "wrongly"

        le = self.df_words.copy()  # [box_cols]

        # then check which other cells are overlapping with that cell...
        # we first calculate the overlap distance for rows & columns
        # and then use that to group them into columns & rows
        le_col_overlap = gu.calc_pairwise_matrix(
            gu.pairwise_box_gap_distance_along_axis_func, le[box_cols].values, diag=-1, axis=0)
        le_row_overlap = gu.calc_pairwise_matrix(
            gu.pairwise_box_gap_distance_along_axis_func, le[box_cols].values, diag=-1, axis=1
        )
        # create row & column clusters with the overlap distance matrix
        # all negative distances mean that two cells overlap, so we set the thresold to 0
        le["cols"], _ = gu.distance_cluster(data=le[box_cols].values,
                                            distance_matrix=le_col_overlap, distance_threshold=0)
        le["rows"], _ = gu.distance_cluster(data=le[box_cols].values,
                                            distance_matrix=le_row_overlap, distance_threshold=0)
        cols = gu.merge_bbox_groups(le, "cols").sort_values(by="x0", ascending=True)
        rows = gu.merge_bbox_groups(le, "rows").sort_values(by="y0", ascending=False)

        # create new column-coordinates in the correct order..
        cols = cols.reset_index()
        rows = rows.reset_index()

        # and map the sorted column & row coordinates to the text boxes
        col_map = pd.Series(cols.index.values, index=cols['index'])
        row_map = pd.Series(rows.index.values, index=rows['index'])
        le["cols"] = le["cols"].map(col_map)
        le["rows"] = le["rows"].map(row_map)
        le = le.sort_values(by=["cols", "rows"])

        # now all we need to do is use the row & column coordinates to create a dataframe!
        table = pd.pivot_table(le, values='text',
                               index=['rows'], columns=['cols'],
                               aggfunc='first', fill_value='')

        # TODO:
        # now create the clusters in order to detect horizonal & vertical lines in
        # the table
        # vlines = gu.cluster1D(x_coordinates.reshape(-1, 1), np.mean, self.tbe.table_line_merge_tol)
        # hlines = gu.cluster1D(y_coordinates.reshape(-1, 1), np.mean, self.tbe.table_line_merge_tol)

        return table, (cols, rows)

    @cached_property
    def df(self):
        """get table as a pandas dataframe"""
        try:
            table, _ = self.convert_cells_to_df()
            table.attrs["source"] = self._filename
            table.attrs["area"] = self.bbox.tolist()
            table.attrs["page"] = self.page
            return table
        except Exception as e:
            logger.error(self.identify_table())
            raise e

    @cached_property
    def metrics(self) -> typing.Dict[str, typing.Any]:
        """standard metrics only consist of the most necessary data
        in order to label pdf tables for parametr optimization"""
        table, (cols, rows) = self.convert_cells_to_df_text_only()
        # TODO: over time calculate better metrics by goind
        #       through our table extractions and check wrong classifications

        # remove some special characters and normalize whitespace
        # to get a good md5 sum for these tables...
        tokentable = table.replace({'<.?s>': '', r'\s+': ' '}, regex=True)
        tablestr = str(tokentable.values.tolist())
        md5 = hashlib.md5(tablestr.encode('utf-8')).hexdigest()

        metrics = dict(
            area_detection_method=self.tbe.extraction_method,
            table_area=self.bbox,
            md5=md5,
            x0=self.bbox[0],
            y0=self.bbox[1],
            x1=self.bbox[2],
            y1=self.bbox[3],
            page=self.page,
            page_bbox=self._page_bbox,
            table=table,
            cols=cols,
            rows=rows,
            file=self._filename
        )

        return metrics

    def identify_table(self):
        return (f"Table {self.bbox} on page {self.page}, "
                f"{self._filename} with "
                f"index: {self.page - 1} can not be extracted.")

    @cached_property
    def is_valid(self):
        # TODO: include all the other table methods to check for validity...
        #       such as filter_correct_finished_tables
        try:
            if any((self.df_le.empty, self.df.size <= 1)):
                return False
            elif self.df_ge.empty:
                logger.warning("TODO: table validation check not yet implemented")
                return True
            else:
                # this function was generated by sklearn decision tree on our test dataset using
                # generate_pdf_table_features.py
                table, (hlines, vlines, cells) = self.convert_cells_to_df()
                words = self.df_words
                table_area = self.bbox

                words["w"] = words.eval("x1-x0")
                words["h"] = words.eval("y1-y0")
                words["area"] = words.eval("w*h")
                words_area_sum = words.area.sum()

                y0 = table_area[1]
                y1 = table_area[3]
                h = (y1 - y0)

                words_area_sum = words_area_sum
                cells_num = table.size  # number of cells in table
                cells_detected_num = len(cells)
                vlines_num = len(vlines)
                h = h

                if "coords" in cells:
                    cells_span_num = (cells.coords.str.len() > 1).sum()
                else:
                    return False

                # this part here is a trained decision tree classifier
                if ((vlines_num / cells_detected_num) + (cells_span_num / cells_detected_num)) <= 1.2666667699813843:
                    if ((words_area_sum / h) - (h / cells_num)) <= 11.474941492080688:
                        return False  # classification scores: [[0.01 0.  ]]
                    else:  # if ((words_area_sum/h)-(h/cells_num)) > 11.474941492080688
                        return True  # classification scores: [[2.20e-02 3.59e+02]]
                else:  # if ((vlines_num/cells_detected_num)+(cells_span_num/cells_detected_num)) > 1.2666667699813843
                    return False  # classification scores: [[0.022 0.   ]]
        except Exception as e:
            logger.error(self.identify_table())
            raise e

    @property
    def metrics_X(self):
        """
        calculate some extended characteristics about the tables
        This helps in situations where we want to find better
        hyperparameters or classifiy tables (e.g. good/bad)...

        TODO: get max density in histogram and use that as a metric
              get the variance of the histogram and use that as a metric...
              tables should hae a pretty low variance, as the elements
              are distributed more equally than with a figure


        """
        try:
            table, (hlines, vlines, cells) = self.convert_cells_to_df()
            words = self.df_words
            table_area = self.bbox

            metrics = self.metrics
            words["w"] = words.eval("x1-x0")
            words["h"] = words.eval("y1-y0")
            words["area"] = words.eval("w*h")
            words_area_sum = words.area.sum()

            x0 = table_area[0]
            y0 = table_area[1]
            x1 = table_area[2]
            y1 = table_area[3]
            w = (x1 - x0)
            h = (y1 - y0)
            area = w * h

            empty_cells = table == ""

            if "text" in cells:
                word_count = cells.text.str.split().str.len().sum()
            else:
                word_count = 0

            metrics.update(dict(
                tableobj=self,
                words_area_sum=words_area_sum,
                word_line_num=words.size,
                word_count=word_count,
                table_area=area,
                table_text_len=len(table.values.sum()),
                cells_num=table.size,  # number of cells in table
                cells_detected_num=len(cells),
                hlines_num=len(hlines),
                vlines_num=len(vlines),
                table_line_count=len(self.df_le),
                table_word_count=len(self.df_words),
                graphic_elem_count=len(self.df_ge),
                x0=x0,
                y0=y0,
                x1=x1,
                y1=y1,
                w=w,
                h=h,
                area=area,
                row_count=table.shape[0],
                col_count=table.shape[1],
                empty_cells_sum=empty_cells.values.sum(),
                empty_cols_count=empty_cells.any(axis=1).sum(),
                empty_rows_count=empty_cells.any(axis=0).sum(),

                # conditional metrics
                cell_detected_size_sum=0,
                cells_span_num=0,  # cells that span more than one cell in the grid
            ))

            if "coords" in cells:
                metrics.update(dict(
                    cell_detected_size_sum=cells.coords.str.len().sum(),
                    cells_span_num=(cells.coords.str.len() > 1).sum(),  # cells that span more than one cell in the grid
                ))
            return metrics
        except Exception as e:
            logger.error(self.identify_table())
            raise e

        ## calculate more statistics about table graphics

        try:
            points = gu.boundarybox_query(tm.pageobj.df_ge, bbox, tol=10)
            a, b = _calculate_ge_points_density_metrics(
                points[["y0", "y1"]].values.flatten())

            # point-density
            hist_resolution = 3  # [px]@72dpi pdf standard resolution
            bins = int(h / hist_resolution)
            ypoints = points[["y0", "y1"]].values.flatten()
            ypoints_hist = np.histogram(ypoints, bins=bins)[0]

            length = np.sqrt(points.w ** 2 + points.h ** 2)
            length_hist = np.histogram(length)

            # find max number of consecutive zeros
            # which equals the largest *gap* in graphic elements coordinates in the histogram
            max_gap = np.diff(np.where(ypoints_hist != 0)[0]).max()
            m2 = pd.Series(dict(
                density_var=b.var(),
                density_norm_var=b.var() / b.mean(),
                density_func=(a, b),
                density_max=b.max(),
                density_min=b.min(),
                density_min_max=b.max() - b.min(),
                density_min_max_norm=(b.max() - b.min()) / b.mean(),
                # density_mean=
                num_points=len(points),
                max_points_bin=ypoints_hist.max(),
                point_bin_var=ypoints_hist.var(),
                point_bin_notnull_var=ypoints_hist[ypoints_hist > 0].var(),
                ypoints_hist=ypoints_hist,
                point_density3=(ypoints_hist > 0).sum() / h,
                point_density=(ypoints_hist > 0).sum() / len(ypoints_hist),
                max_gap=max_gap,
                max_gap_perc=max_gap / ypoints_hist.size
                # len_group_density = point_groups.iloc[0]/row.area*100,
                # distribution = len(point_groups)/len(points),
                # max_regularity = point_groups.iloc[0]/len(points),
                # max3dist = point_groups.iloc[:3].sum()/len(point_groups)
            ))  # .mul(100).astype(int)

            return pd.concat([tm, m1, m2])
        except:  # something didn't work maybe there are no graphics elements?
            return pd.concat([tm, m1])


def filter_out_small_graphics_elements(
        ge: pd.DataFrame,
        max_area_page_ratio, page_area, margin,
        page_bbox, min_elem_x, min_elem_y
) -> pd.DataFrame:
    """get filtered graph elements by throwing out a lot of unnecessary graphics
    elements which are probably not part of a table

    # TODO: parameterize this functions for hyperparameter tuning...

    we want to filter out a lot of graphics element which are not relevant. Those include:
    - very small elements
    - elements far away from text
    - elements which are are not a box or a vertical/horizontal line
    # TODO: check out of we can filter out even more than this ;)

    check out if we a lot of elements with "similar" x- or y coordinates in
    order to filter out "non-ordered" elements which
    we assume to be non-table graphical elments

    TODO: some of the following operations might be more efficient in the table-graph-search
    """

    # TODO: filter specifically for each coordinate
    # TODO: filter for "unique-coordinate-density"
    #       as tables are ordered, we can assume that if we
    #       seach for "unique" coordinates, the density of
    #       tables should be even lower than that for figures.
    #       because of this we can theoretically improve the
    #       accuracy of this filter.

    # check if the following calculations are duplicated anywhere...
    ge['w'] = ge.x1 - ge.x0
    ge['h'] = ge.y1 - ge.y0
    ge['area'] = ge.w * ge.h
    ge['area_ratio'] = (ge.area / page_area)
    # df_ge['w+h'].hist(bins=100)
    # filter out all elements that occupy more than max_area_page_ratio of page space
    ge_v = ge[ge.area_ratio < max_area_page_ratio]
    # filter out all elements that are thinner than min_elem_x- and y
    # under certain conditions
    ge_v = ge_v.query('((w>h) and (w>@min_elem_x)) or ((h>w) and (h>@min_elem_y))').copy()
    # elements that are "outside the page area/margin should be discarded
    # "standard" margin
    ge_v = ge_v[((ge_v[box_cols[2:]] + margin) < (page_bbox[2:])).all(1)
                & ((ge_v[box_cols[:2]] - margin) > (page_bbox[:2])).all(1)]
    # use the smallest rectangle which encloses textboxes plus a small margin
    # min_x = self.box_groups.x0.min()
    # max_x = self.box_groups.x1.max()
    # min_y = self.box_groups.y0.min()
    # max_y = self.box_groups.y1.max()
    # ge_v = ge_v[((ge_v.x0 + margin) > min_x) & ((ge_v.x1 - margin) < max_x)
    #             & ((ge_v.y0 + margin) > min_y) & ((ge_v.y1 - margin) < max_y)]

    # elements

    return ge_v


class TableCandidateAreasExtractor(Operator):
    """produces a list of potential table objects"""

    def __init__(
            self,
            table_extraction_params: TableExtractionParameters = None,
            method="pdf",
    ):
        super().__init__()
        self._method = method
        self._tbe = table_extraction_params or TableExtractionParameters.reduced_params()

    def use_table_transformer(self, img: PIL.Image.Image, pages_bbox: np.ndarray):
        from transformers import AutoImageProcessor, TableTransformerForObjectDetection
        import torch

        image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
        model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

        inputs = image_processor(images=img, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        target_sizes = torch.tensor([img.size[::-1]])
        results = image_processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0]

        pdf_size = pages_bbox[2:]
        rendersize = img.size
        ratio = pdf_size / rendersize

        areas_raw = results['boxes'].detach().numpy()
        areas = pd.DataFrame(areas_raw * ratio[0], columns=['x0', 'y0', 'x1', 'y1'])
        # areas = (2*rendersize-areas)*ratio[0]
        # np.array([size[1],0,0,0])-areas
        areas[['y1', 'y0']] = pdf_size[1] - areas[['y0', 'y1']]

        # timg = img.crop(areas_raw[0] + (-margin, -margin, margin, margin))
        return areas

    def use_pdf_source(
            self,
            graphic_elements: list[pydoxtools.document_base.DocumentElement],
            line_elements: list[pydoxtools.document_base.DocumentElement],
            pages_bbox,
            text_box_elements: list[pydoxtools.document_base.DocumentElement],
            filename=None
    ):
        # get minimum length for lines by searching for
        # the minimum height/width of a text box
        # we do this, because we assume that graphical elements should be at least this
        # big in order to be part of a table
        # TODO: make all of these parameters configurable
        #       so that we can train them as hyperparameters
        min_size = 5.0  # minimum size of a graphics element
        margin = 20  # margin of the page
        max_area_page_ratio = 0.4  # maximum area on a page to occupy by a graphics element
        ge = pd.DataFrame(graphic_elements)
        le = pd.DataFrame(line_elements)
        text_box_elements = pd.DataFrame(text_box_elements)
        pages = ge.p_num.unique()
        # we keep distance_threshold constant as the same effect can be gained
        # through tbe.area_detection_params but a lot more fine-grained as
        # it directly controls the sensitivity of the distance function
        # TODO: use this as a parameter in table_extraction_parameters?
        #       might not make sense as it is the same as tbe.area_detection_params
        # merge everything with a distance of less than 10..
        distance_threshold = 10.0  # for table area candidates (TODO: parameterize?)

        # detect table areas page-wise
        box_iterations: dict[int, list[pd.DataFrame]] = {}
        table_candidates: list[PDFTableCandidate] = []
        for p in pages:
            tbe = text_box_elements.loc[text_box_elements.p_num==p].copy()
            # do some calculations
            # tbe['y_mean'] = tbe[['y0', 'y1']].mean(axis=1)
            # tbe['x_mean'] = tbe[['x0', 'x1']].mean(axis=1)
            tbe['w'] = tbe.x1 - tbe.x0
            tbe['h'] = tbe.y1 - tbe.y0
            min_elem_x = max(tbe.w.min(), min_size)
            min_elem_y = max(tbe.h.min(), min_size)
            page_bbox = b = pages_bbox[p]
            page_area = b[2] * b[3]  # we can do this because the bounding box is always (0,0) at lower left
            df_ge = filter_out_small_graphics_elements(
                ge=ge[ge["p_num"] == p].copy(), max_area_page_ratio=max_area_page_ratio,
                page_area=page_area, margin=margin,
                min_elem_x=min_elem_x, min_elem_y=min_elem_y,
                page_bbox=page_bbox
            )
            df_le = le[le["p_num"] == p]
            # TODO: make TableExtractionParameters configurable in document
            table_areas, box_iterations[p] = detect_table_area_candidates(
                self._tbe,
                df_le, df_ge,
                distance_threshold
            )
            _table = (
                PDFTableCandidate(
                    df_le, df_ge,
                    initial_area=row[box_cols],
                    page_bbox=page_bbox, page=p, file_name=filename
                ) for _, row in table_areas.iterrows()
            )
            table_candidates.extend(t for t in _table if not t.df_le.empty)

        return dict(
            table_candidates=table_candidates,
            box_levels=box_iterations
        )

    def __call__(
            self,
            graphic_elements: list[pydoxtools.document_base.DocumentElement],
            line_elements: list[pydoxtools.document_base.DocumentElement],
            pages_bbox,
            text_box_elements: list[pydoxtools.document_base.DocumentElement],
            filename=None,
            images: dict[int, PIL.Image.Image] = None,
    ):
        # TODO: merge the common parts of the "use" method
        if self._method == "images":
            table_areas = []
            df_le = pd.DataFrame(line_elements)
            df_ge = pd.DataFrame(graphic_elements)
            pages = df_le.p_num.unique()
            for page_num in pages:
                img = images[page_num]
                areas = self.use_table_transformer(img, pages_bbox[page_num])
                for area in areas.values:
                    table = PDFTableCandidate(
                        df_le, df_ge, area,
                        tbe=self._tbe,
                        page=page_num, page_bbox=pages_bbox[page_num],
                        file_name=filename)
                    table_areas.append(table)

            return dict(
                table_candidates=table_areas,
                box_levels=[]
            )
        else:
            return self.use_pdf_source(graphic_elements,
                                       line_elements,
                                       pages_bbox,
                                       text_box_elements,
                                       filename)


def detect_table_area_candidates(
        tbe: TableExtractionParameters,
        df_le, df_ge,
        distance_threshold: float
):
    """
    Detect tables from elements such as textboxes & graphical elements.
    the function expects a range of parameters which need to be tuned.

    TODO: sort out non-table area regions after every iteration and speed up subsequent
          table search iterations this way.. But optimize this on a recall-basis
          in order to make sure we don't sort out any valid tables...
    returns:

    (table_groups, box_iterations):
        box_iterations: for debugging, save the progression on how boxes were merged
        step-by-step at every iteration
    """
    boxes = pd.concat([
        df_le[box_cols] if (
                "t" in tbe.extraction_method and (not df_le.empty)) else pd.DataFrame(),
        df_ge[box_cols] if (
                "g" in tbe.extraction_method and (not df_ge.empty)) else pd.DataFrame(),
    ])
    if len(boxes) == 0:
        return pd.DataFrame(), []

    box_iterations = []

    # TODO: if (graphic) boxes are empty, revert to text-based...
    # TODO: do this in several (configurable) iterations
    if not tbe.area_detection_distance_func_params:
        raise ValueError("no area_detection_distance_func_params defined!")
    if len(boxes) > 1:  # merge boxes to table areas..
        for level, param_level in enumerate(tbe.area_detection_distance_func_params):
            x = gu.calc_pairwise_matrix(
                gu.pairwise_weighted_distance_combination, boxes.values, diag=0,
                parameter_list=param_level
            )

            boxes["groups"], dist_m = gu.distance_cluster(
                distance_matrix=x, distance_threshold=distance_threshold
            )
            # create new column with the type of group (hb,hm,ht,vb,vm,vt) and their labels
            boxes = gu.merge_bbox_groups(boxes, "groups")
            box_iterations.append(boxes)
            if len(boxes) < 2:
                break

    # TODO: check variance to sort out "bad" columns as an additional parameter?
    #       but maybe it would also be a good idea to simply do that during the distance calculation
    # line_groups.groupby('vh_left_top_group').x0.apply(lambda x: x.var())

    # filter out areas without any text
    # we only have a valid table if there is actualy text to be processed...
    # TODO:  also handly tables with figures only at some point in the future?
    # TODO: should we sort out more "bad" areas here already? may speed up table extraction...
    # boundarybox_intersection_query(bbs=df_le, bbox=box)
    # filter our empty groups
    # TODO: right now, we don't really know what would be a good filter...
    #       maybe do this by using an optimization approach
    text_cell_num = boxes[box_cols].apply(lambda x: len(boundarybox_intersection_query(bbs=df_le, bbox=x)), axis=1)
    boxes[text_cell_num > 0].copy()
    table_groups: pd.DataFrame = _filter_boxes(
        boxes,
        min_area=tbe.min_table_area,
        min_aspect_ratio=tbe.min_aspect_ratio,
        max_aspect_ratio=tbe.max_aspect_ratio
    )

    # sort table candidates according to y-coordinates top-to-bottom
    table_groups = table_groups.sort_values(
        by=["y1", "x0", "y0", "x1"], ascending=[False, True, False, True])

    return table_groups, box_iterations


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


class Iterator2Dataframe(pydoxtools.operators_base.Operator):
    """convert arbitrary iterators with arguments into pandas dataframes"""

    def __call__(self, iterator):
        def create_data_frame(*args, **kwargs):
            df = pd.DataFrame(iterator(*args, **kwargs))
            return df

        return create_data_frame
