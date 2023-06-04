"""
This file gathers functions that are used multiple times
in our jupyter notebooks...  They are not intended for production
use.
"""

from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import concurrent.futures
import concurrent.futures
import functools
import itertools
import logging

import numpy as np
import pandas as pd
from tqdm import tqdm

import pydoxtools.cluster_utils as gu
from pydoxtools import pdf_utils, html_utils, file_utils
from pydoxtools.cluster_utils import box_cols
from pydoxtools.settings import settings

logger = logging.getLogger(__name__)

box_cols = gu.box_cols

logger = logging.getLogger(__name__)
tqdm.pandas()


def get_unique_pdf_file(pdf_link: str, url: str) -> str:
    new_pdf_link = html_utils.absolute_url_path(url, pdf_link)
    return file_utils.generate_unique_pdf_filename_from_url(new_pdf_link)


# TODO: make it possible to find more than just one neighbouring area
def find_closest_match_idx(table, table_list, k=1):
    """finds the closest match to "table" in terms of detected area"""
    file_labels = table_list.query('file==@table.file')
    page_mul = int(1e7)
    query = table[box_cols + ['page']].astype(int).values * [1, 1, 1, 1, page_mul]
    data = file_labels[box_cols + ['page']].astype(int).values
    data[:, 4] *= page_mul
    # we are only interested in the closest match
    res = gu.distance_query_manhattan(query, data, k=k)

    if len(res) == 0:
        return pd.Series((np.NAN, np.NAN))
    # get original area index
    res_idx = res[0, 1]
    res_dist = res[0, 0]
    idx = file_labels.iloc[res_idx].name
    return pd.Series((idx, res_dist))


def extract_tables(pdf_file, table_extraction_params, cached=False):
    # pdfi = pdf_utils.repair_pdf_if_damaged(pdf_utils.extract_all_infos)(
    # TODO: for actual optimization we need to replace below with a non-cached version
    # pdfi = pdf_utils.PDFDocumentOld.from_disk_cache(pdf_file,
    # pdfi = pdf_utils.PDFDocumentOld.pre_initialized(pdf_file,
    initfunc = pdf_utils.PDFDocumentOld.with_cached_elements
    if cached:
        initfunc = pdf_utils.PDFDocumentOld.from_disk_cache
    try:
        pdf = initfunc(pdf_file,
                       page_numbers=None,
                       table_extraction_params=table_extraction_params
                       )
    except:
        logger.exception(f"something went wrong with file {pdf_file}")
        raise Exception(f"something went wrong with file {pdf_file}")
    return pdf.table_metrics_X


def detect_tables(
        files, table_extraction_params,
        max_filenum=20,
        cached=False,
):
    pdfs = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        # with concurrent.futures.ThreadPoolExecutor(1) as executor: # for debugging purposes
        for fn, pdfi in zip(files, tqdm(executor.map(
                # pdf_utils.get_pdf_text_safe_cached, files[:max_filenum]
                extract_tables,
                files[:max_filenum],
                itertools.repeat(table_extraction_params),
                itertools.repeat(cached)
        ))):
            pdfs.append(pdfi)

    # TODO: do the following in parallel!
    tables = pd.concat([tm for tm in tqdm(pdfs)])
    tables.file = tables.file.str.split("/").str[-1]
    return tables.reset_index().drop(columns='index')


md5_cols = ["md5", "page", "file"]


@functools.lru_cache
def load_tables_from_csv():
    """TODO: move this into an extractor!!"""
    table_data = settings.TRAINING_DATA_DIR / "pdfs/tabledata.csv"

    # match labeled tables with calculated data
    tables_csv = pd.read_csv(table_data)
    # remove intermediate header rows...
    tables_csv = tables_csv[tables_csv.file != "file"]
    # convert values to actual numeric types
    tables_csv[gu.box_cols] = tables_csv[gu.box_cols].astype(float).astype(int)
    tables_csv.page = tables_csv.page.astype(float).astype(int)
    tables_csv.labeling_time = pd.to_datetime(tables_csv.labeling_time)
    # keep only the last labeled version of the same table area and sme table text (md5)
    # keep only the last labeled version of the same table text..
    tables_csv = tables_csv.sort_values(by='labeling_time') \
        .drop_duplicates(subset=["file", "page", "x0", "y0", "x1", "y1"], keep='last') \
        .drop_duplicates(subset=md5_cols, keep='last')
    return tables_csv


def merge_table_classifications(tables, tables_csv):
    good = ['all_good', 'all_good-']

    # merge the 100% redetected tables based on md5 sum
    tables_md5_matched = tables_csv.reset_index().merge(
        tables.reset_index(),  # only use the files that are also in tables
        on=md5_cols,
        indicator=True, how="inner", suffixes=(None, "_detected")
    ).set_index("index").rename(columns={'index_detected': 'match_idx_md5'})[['match_idx_md5']]
    print(f"merged on: {md5_cols}")
    print(tables_md5_matched.columns, tables_md5_matched.shape)

    tables_csv_matched = tables_csv.join(tables_md5_matched)
    print(tables_csv_matched.shape, tables_csv.shape)

    # find detected tables that are close to tables that were at some point labeled as well..
    tables_csv_matched[['match_idx_bbox', 'match_bbox_dist']] = tables_csv.progress_apply(
        lambda x: find_closest_match_idx(x, tables), axis=1)

    # use area match on tables taht weren't matched using md5 yet

    tables_csv_matched['match_idx'] = tables_csv_matched['match_idx_md5']
    maxpixdiff = 3
    allowed_area_match = tables_csv_matched.eval(
        "match_idx.isnull() & (match_bbox_dist<=(4*@maxpixdiff)) & ~classification.isin(@good)")
    tables_csv_matched.loc[
        allowed_area_match, 'match_idx'
    ] = tables_csv_matched.loc[allowed_area_match, 'match_idx_bbox']
    print(tables_csv_matched.shape)

    # now drop all csv-tables that match with the same detected table within a certain margin
    # as the csv table is already sorted after dates, the last one automatically resembles the last
    # labeling

    # tables_csv_matched.drop(drop_tables)
    drop_tables = tables_csv_matched.query(
        "match_idx_bbox.duplicated(keep='last') & (match_bbox_dist<=(4*@maxpixdiff)) & match_idx_md5.isnull()")
    tables_csv_matched = tables_csv_matched.drop(drop_tables.index)
    print(tables_csv_matched.shape)

    # drop all area_matched tables that were already matched by md5

    md5_idx_unique = tables_csv_matched["match_idx_md5"].unique()
    drop_tables = tables_csv_matched.query(
        "~match_idx_bbox.isin(@md5_idx_unique) & (match_bbox_dist<=(4*@maxpixdiff)) & match_idx_md5.isnull()")
    tables_csv_matched = tables_csv_matched.drop(drop_tables.index)
    print(tables_csv_matched.shape)

    #### merge labeled tables
    # finally, merge the labeled table classes with the detected ones based on match_idx

    tables_merged = tables.reset_index().merge(
        # tables_csv.query("_merge=='right_only'").drop(columns='_merge'),  # only use the files that are also in tables
        tables_csv_matched, left_index=True,
        right_on=["match_idx"],
        indicator=True, how="outer", suffixes=(None, "_csv")
    ).reset_index()
    return tables_merged
