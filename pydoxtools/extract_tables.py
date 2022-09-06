import functools
import hashlib
import typing
from functools import cached_property
from typing import Any

import numpy as np
import pandas as pd
import pydantic
from pdfminer.layout import LTChar
from sklearn.neighbors import KernelDensity

from pydoxtools import cluster_utils as gu
from pydoxtools.cluster_utils import pairwise_txtbox_dist, box_cols, y1, x0, x1
from pydoxtools.document import Extractor
from pydoxtools.extract_textstructure import _line2txt


class TableExtractionParameters(pydantic.BaseModel):
    # possible values: "t", "g", "tg", with "t" for text-based and "g" for graphics based
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


class TableExtractor(Extractor):
    def __init__(self, table_extraction_params: typing.Optional[TableExtractionParameters] = None):
        self._table_extraction_params = table_extraction_params

    @property
    def list_lines(self):
        return []

    @property
    def tables(self) -> list[dict[str, dict[str, Any]]]:
        """
        table in the following (row - wise) format:

        [{index -> {column -> value } }]
        """
        return []

    @property
    def tables_df(self) -> List["pd.DataFrame"]:
        return []


def _calculate_ge_points_density_metrics(points):
    """calculate vertical point density (edge density would be without vertical lines)

    a,b = _calculate_ge_points_density_metrics(points[["y0","y1"]].values.flatten())
    pd.DataFrame([a,b]).T.plot(x=0,y=1)
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


def _LTObj2Chars(df_le) -> pd.DataFrame:
    chars = []
    for text_line in df_le.lineobj.values:
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
        # check if we have a top element that somehow exists within the cell borders
        # and can be used to close the cell...
        # we have 3 cases:
        #  - both ends are inside the cell
        #  - x0 is on the left side of the left lineend
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


class PDFTable:
    def __init__(self, parent_page: PDFPageOld, initial_area: np.ndarray):
        self.parent_page: PDFPageOld = parent_page
        self.initial_area: np.ndarray = initial_area

        self.max_lines = 1000

        self._debug = {}

    @property
    def tbe(self) -> TableExtractionParameters:
        return self.parent_page.tbe

    @cached_property
    def df_le(self) -> pd.DataFrame:
        """line elements of table"""
        return gu.boundarybox_query(
            self.parent_page.df_le, self.initial_area,
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
        """Calculate word boxes instead of line boxes for tables
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
    def df_ge(self) -> pd.DataFrame:
        """graphic elements of table"""
        # TODO: maybe we should not use filtered, but unfiltered graphic elements here?
        return gu.boundarybox_query(
            self.parent_page.df_ge_f, self.initial_area,
            tol=self.tbe.text_extraction_margin
        ).copy()

    @cached_property
    def bbox(self):
        # find the exact bounding box of our table...
        ge_bb = np.array([self.df_ge.x0.min(), self.df_ge.y0.min(), self.df_ge.x1.max(), self.df_ge.y1.max()])
        le_bb = np.array([self.df_le.x0.min(), self.df_le.y0.min(), self.df_le.x1.max(), self.df_le.y1.max()])
        return np.array([*np.vstack([ge_bb[:2], le_bb[:2]]).min(0), *np.vstack([ge_bb[2:], le_bb[2:]]).max(0)])

    @functools.lru_cache()
    def detect_cells(self, steps=None) -> pd.DataFrame:
        """
        This algorithm works by slowly scanning through a table bottom-to-top first
        and left-to-right for each row.

        On the way we detect rows by alternating between text lines and graphical lines.
        We can detect a cell border by going from the bottom of a text-line-box to the botom
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
        ge = self.df_ge[box_cols + ['w', 'h']].sort_values(
            by=['y0', 'y1', 'x0', 'x1'],
        ).set_index(['y0', 'y1', 'x0', 'x1'], drop=False).copy()
        le = self.df_words[box_cols + ['text']].sort_values(
            by=["y0", "x0", "y1", "x1"],
        ).set_index(['y0', 'x0', 'y1', 'x1'], drop=False).copy()

        # generate a list of increasing y-mean values which we can "pop"
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

        max_steps = steps or self.max_lines  # maximum of 1000 lines for a table
        for i in range(max_steps):
            if not y_mean_list:  # if y_mean_list is empty, we reached the top of the table
                # as there are no more text lines we can close up all cells
                # that are still open
                last_cells = [_close_cell(oc, le, self.bbox[y1], text_line_tol) for oc in open_cells]
                cells.extend(lc for lc in last_cells if lc)
                break

            # advance y0_cursor to the next textline
            # TODO: use y_mean to the the left_over_lines!!
            while y_mean_list:  # do this until we don't have any more text elements left
                y0_cursor = y_mean_list.pop()
                # make sure that the middle of the line is above the previously found table row
                if y0_cursor > y0_h_elem:
                    break

            # now we would like to know every vertical line (not element) that crosses the current
            # y0-cursor-line to get the vertical cell borders in this row...
            # get all elements reaching from `below` to `above + tolerance` (using y0 and y1)
            # we do "swaplevel" in order to be able to use more efficient x-indexing afterwards
            row_ge_elem = ge.loc[idx[:y0_cursor + elem_scan_tol, y0_cursor + elem_scan_tol:], :]
            v_row_elem = row_ge_elem.swaplevel(0, 2, axis=0).sort_index()

            if v_row_elem.empty:
                # no vertical line detected in this row, so we should move further upwards
                # and set y0_h_elem to the top of this line element.
                if y_mean_list:
                    y0_h_elem = (y0_cursor + y_mean_list[-1]) / 2
                continue

            # get all vertical coordinates because we would like to identify the lines and not
            # boxes...
            vlines = np.sort(np.hstack(
                (np.unique(v_row_elem[['x0', 'x1']].values), self.bbox[[x0, x1]])))
            x0_cursor = vlines[0]
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

            # now after scanning the row, advance y-cursor upwards and check which cells we can close...
            # get the next horizontal element
            # TODO: handle the case where we have text "above" graphic lines...
            # TODO: might need to check for y1 here as well, as there
            #       might be a case were the box only closes and no new
            #       one opens... more tests will show...
            #       maybe check whatever is between here and the next text element?
            #       and then take the minimum y-coordinate from that..
            #       YEAH --> we need this.. already our first test showed :P
            left_over_y1 = ge.loc[idx[:, y0_cursor:], :]
            if left_over_y1.empty:
                # assume we are in the last box and need to
                # close up any remaining boxes...
                continue
            left_over_y0 = left_over_y1[y0_cursor:]
            next_h_elem = left_over_y1.iloc[0]
            # TODO: do we need a min here?
            y0_h_elem = next_h_elem.y1 if left_over_y0.empty else min(next_h_elem.y1, left_over_y0.iloc[0].y0)
            # TODO: we might want to check if *max_v_line_thickness* should be the
            #       distance to the next text element.... but maybe its enough to simply
            #       have this as a parameter...
            # select horizontal elements in row
            h_row_elem = left_over_y1.loc[
                         idx[: y0_h_elem + max_v_line_thickness,
                         y0_h_elem - elem_scan_tol:],
                         :]
            h_row_elem = h_row_elem.loc[
                h_row_elem.w > min_cell_width]  # .reorder_levels([2, 3, 0, 1], axis=0).sort_index()
            # and extract horizontal lines from them
            h_lines = pd.concat([
                h_row_elem.loc[y0_h_elem - elem_scan_tol:y0_h_elem + max_v_line_thickness,
                ["x0", "x1", "y0"]].rename(columns={"y0": "y"}),
                h_row_elem.loc[idx[:, y0_h_elem - elem_scan_tol:y0_h_elem + max_v_line_thickness],
                               ["x0", "x1", "y1"]].rename(columns={"y1": "y"})
            ]).droplevel(["y0", "y1"]).sort_index().drop_duplicates(["x0", "x1"])

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

    @functools.lru_cache()
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

        # first thing we need to do is to cluster cell borders
        x_coordinates = cells[["x0", "x1"]].values
        y_coordinates = cells[["y0", "y1"]].values

        vlines = gu.cluster1D(x_coordinates.reshape(-1, 1), np.mean, self.tbe.table_line_merge_tol)
        hlines = gu.cluster1D(y_coordinates.reshape(-1, 1), np.mean, self.tbe.table_line_merge_tol)

        xtcs = _get_cell_coordinates(x_coordinates, vlines, tol=self.tbe.cell_idx_tol)
        ytcs = _get_cell_coordinates(y_coordinates, hlines, tol=self.tbe.cell_idx_tol)

        # --> sometimes we create "weired" vertical/horzontal lines by averaging them.
        # in that case some cells don't fit into the raste anymore and we have too few lines for every
        # cell. usually this happens in areas that are not real tables such as figures...
        # so this is a rally nice way to sort som of them out ;).
        # TODO: maybe at some poin in the future we need a more robust method to create
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

    @cached_property
    def df(self):
        """get table as a pandas dataframe"""
        try:
            table, _ = self.convert_cells_to_df()
            return table
        except Exception as e:
            logger.error(self.identify_table())
            raise e

    @cached_property
    def metrics(self) -> typing.Dict[str, typing.Any]:
        """standard metrics only consist of the most necessary data
        in order to label pdf tables for parametr optimization"""
        table, (hlines, vlines, cells) = self.convert_cells_to_df()
        # TODO: over time calculate better metrics by goind
        #       through our table extractions and check wrong classifications

        # remove some special characters and normalize whitespace
        # to get a good md5 sum for these tables...
        tokentable = table.replace({'<.?s>': '', '\s+': ' '}, regex=True)
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
            page=self.parent_page.pagenum,
            page_bbox=self.parent_page.page_bbox,
            table=table,
            file=self.parent_page.parent_document.filename
        )

        return metrics

    def identify_table(self):
        return (f"Table {self.bbox} on page {self.parent_page.pagenum}, "
                f"{self.parent_page.parent_document.filename} with "
                f"index: {self.parent_page.pagenum - 1} can not be extracted.")

    @cached_property
    def is_valid(self):
        # TODO: include all the other table methods to check for validity...
        #       such as filter_correct_finished_tables
        try:
            if any((self.df_le.empty,
                    self.detect_cells().empty,
                    self.df.size <= 1)):
                return False
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
                pageobj=self.parent_page,
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
                empty_cols_count=empty_cells.any(1).sum(),
                empty_rows_count=empty_cells.any(0).sum(),

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


class TableExtractionError(Exception):
    pass
