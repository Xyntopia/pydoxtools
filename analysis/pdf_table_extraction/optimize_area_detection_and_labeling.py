# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown] tags=[]
# # optimize table area detection and label 
#
# we take our algorithm parameters here and try to optimize the parameters to be able to
# detect as many table areas as possible

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

import concurrent.futures
import itertools
import logging
import sys

import numpy as np
import optuna
import pandas as pd
import torch
from IPython.display import display
from matplotlib import pyplot as plt
from tqdm import tqdm
import random
from IPython.display import display, HTML

import pydoxtools.extract_tables
import pydoxtools.visual_document_analysis as vda
from pydoxtools import cluster_utils as gu
from pydoxtools import nlp_utils
from pydoxtools import pdf_utils, file_utils
from pydoxtools.settings import settings
from pydoxtools.labeling import find_closest_match_idx

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
logging.getLogger('readability.readability').setLevel(logging.WARNING)

box_cols = pdf_utils.box_cols

tqdm.pandas()

pdf_utils._set_log_levels()
memory = settings.get_memory_cache()

nlp_utils.device, torch.cuda.is_available(), torch.__version__


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))


def isnotebook():
    try:
        shell = get_ipython().__class__.__name__
        if shell == 'ZMQInteractiveShell':
            return True  # Jupyter notebook or qtconsole
        elif shell == 'TerminalInteractiveShell':
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False


# %% [markdown]
#
#
# ## load pdf files

# %%
settings.TRAINING_DATA_DIR

# %%
files = file_utils.get_nested_paths(settings.PDFDIR, "*.pdf")
files = file_utils.get_nested_paths(settings.TRAINING_DATA_DIR / "pdfs/datasheet", "*.pdf")
# create a database so that we can lookup the actual paths fomr jsut the filenames
pathdb = {f.name: f for f in files}
len(files)

# %%
# pdf_utils.PDFDocumentOld.from_disk_cache.clear()

# %%
import warnings

warnings.filterwarnings('ignore')


def extract_tables(pdf_file, table_extraction_params):
    # pdfi = pdf_utils.repair_pdf_if_damaged(pdf_utils.extract_all_infos)(
    # TODO: for actual optimization we need to replace below with a non-cached version
    # pdfi = pdf_utils.PDFDocumentOld.from_disk_cache(pdf_file,
    # pdfi = pdf_utils.PDFDocumentOld.pre_initialized(pdf_file,
    try:
        pdf = pdf_utils.PDFDocumentOld.with_cached_elements(pdf_file,
                                                            page_numbers=None,
                                                            table_extraction_params=table_extraction_params
                                                            )
    except:
        logger.exception(f"something went wrong with file {pdf_file}")
        raise Exception(f"something went wrong with file {pdf_file}")
    return pdf.table_metrics_X


def detect_tables(
        files, table_extraction_params,
        max_filenum=20
):
    pdfs = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
    #with concurrent.futures.ThreadPoolExecutor(1) as executor: # for debugging purposes
        for fn, pdfi in zip(files, tqdm(executor.map(
                # pdf_utils.get_pdf_text_safe_cached, files[:max_filenum]
                extract_tables,
                files[:max_filenum],
                itertools.repeat(table_extraction_params)
        ))):
            pdfs.append(pdfi)

    # TODO: do the following in parallel!
    tables = pd.concat([tm for tm in tqdm(pdfs)])
    tables.file = tables.file.str.split("/").str[-1]
    return tables.reset_index().drop(columns='index')


# %% [markdown] tags=[]
# ## load labeled tables from csv

# %% tags=[]
table_data = settings.TRAINING_DATA_DIR / "pdfs/tabledata.csv"

# %%
# this should ALWAYS be predefined in order to keep the csv consistent
save_columns = sorted(['area_detection_method',
                       'classification', 'file', 'labeling_time',
                       'md5', 'page', 'x0', 'x1', 'y0', 'y1'])
int_cols = ["page", "x0", "x1", "y0", "y1"]
md5_cols = ["md5","page","file"]

# %% tags=[]
# match labeled tables with calculated data
tables_csv = pd.read_csv(table_data)
# remove intermediate header rows...
tables_csv = tables_csv[tables_csv.file != "file"]
# convert values to actual numeric types
tables_csv[pdf_utils.box_cols] = tables_csv[pdf_utils.box_cols].astype(float).astype(int)
tables_csv.page = tables_csv.page.astype(float).astype(int)
tables_csv.labeling_time = pd.to_datetime(tables_csv.labeling_time)
# keep only the last labeled version of the same table area and sme table text (md5)
# keep only the last labeled version of the same table text..
tables_csv = tables_csv.sort_values(by='labeling_time') \
    .drop_duplicates(subset=["file", "page", "x0", "y0", "x1", "y1"], keep='last') \
    .drop_duplicates(subset=md5_cols, keep='last')
print(tables_csv.columns)

# %% [markdown] tags=[]
# ## manually optimize and generate tabl statistics
#
# we need this, if we want to manually adapt the parameters in oder to label functions...

# %%
# manual file list for labeling... 
manual_file = [
    settings.TRAINING_DATA_DIR / "pdfs/datasheet/14-DD05A.08(II)_DE_TSM_DD05A_08_II_plus_datasheet_B_2017_web.4f.pdf"]
files[0]

# %%
max_files = 100
tabfiles = files#random.sample(files,max_files)
hp = {'es1': 11.1, 'es2': 2.1, 'gs1': 11.1, 'gs2': 20.1}
adp = [{
    "va": [hp['gs1'], hp['es1'], hp['es1'] / 2, hp['es1']],
    "ha": [hp['gs1'], hp['es1'], hp['es1'] / 2, hp['es1']]
},
{
    "va": [hp['gs2'], hp['es2'], hp['es2'] / 2, hp['es2']],
    "ha": [hp['gs2'], hp['es2'], hp['es2'] / 2, hp['es2']]
}]
ex_params = pydoxtools.extract_tables.TableExtractionParameters(
    area_detection_distance_func_params=adp,
    text_extraction_margin=20.0
)

if isnotebook() or getattr(sys, 'gettrace', None):
    tables = detect_tables(
        tabfiles, table_extraction_params=ex_params,
        max_filenum=-1
    )

# %% [markdown] tags=[]
# ## calculate statistics (Tp,Fp,Tn,Fn), unlabeled

# %%
print(tables_csv.classification.unique())
# "area_correct_checked" is the label where the text might be correct, but we didn't check yet..
# in other words, it has to be relabeled
good_area = ['almost correct', 'area_correct_checked', 'area_correct_checked_hf', 'area_correct', 'all_good',
             'area_correct_with_header_footer']


# %%
def merge_labels_by_closest_area(tables, tables_csv):
    match_cols = ["match_idx", "match_dist"]
    tables_csv_selected = tables_csv.query('file.isin(@tables.file.unique())')
    tables[match_cols] = \
        tables.progress_apply(lambda x: find_closest_match_idx(x, tables_csv), axis=1)
    tables_merged = tables.merge(
        tables_csv_selected,  # only use the files that are also in tables
        left_on=["match_idx"], right_index=True,
        indicator=True, how="outer", suffixes=(None, "_csv")
    ).reset_index()
    return tables_merged

if isnotebook():
    tables_merged = merge_labels_by_closest_area(tables, tables_csv)

# %% [markdown]
# TODO: so one problem w have right now is that we should try to mark "good" tables that were found in  similar spot as duplicates
#       because this will generally lower our measurement of the performance of the algorithm... if we have several "good" tables
#       ...  but dies it matter? --> in the end only the md5 counts... 

# %%
if isnotebook():
    tol = 4 * 5  # maximum of XX pixels per edge on average in order to find a match
    good_matches = tables_merged.query("match_dist<@tol")  # areas that were probably already classified
    bad_matches = tables_merged.query(
        "match_dist>=@tol | match_dist.isna()")  # areas that were probably not classified yet
    Tp = good_matches.query("classification.isin(@good_area)")
    Fp = good_matches.query("~classification.isin(@good_area)")
    Tn = bad_matches.query("~classification.isin(@good_area)")
    Fn = bad_matches.query("classification.isin(@good_area)")
    tp, fp, tn, fn = len(Tp), len(Fp), len(Tn), len(Fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = 0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)
    unlabeled = tables_merged.query("match_dist>=@tol")
    stats = dict(
        total=len(tables_merged),
        tp=tp, fp=fp,
        tn=tn, fn=fn,
        precision=np.round(precision, 2),
        recall=np.round(recall, 2),
        F1=np.round(F1, 3),
        unlabeled=len(unlabeled),
    )
    print(stats)


# %% [markdown] jp-MarkdownHeadingCollapsed=true tags=[]
# ### current state
#
#  {'total': 835,
#  'tp': 67,
#  'fp': 116,
#  'tn': 616,
#  'fn': 36,
#  'precision': 0.37,
#  'recall': 0.65,
#  'F1': 0.469,
#  'unlabeled': 11}
#  
#  {'total': 742, 'tp': 32, 'fp': 11, 'tn': 635, 'fn': 64, 'precision': 0.74, 'recall': 0.33, 'F1': 0.46, 'unlabeled': 33}
#  
#  {'total': 577, 'tp': 20, 'fp': 9, 'tn': 511, 'fn': 37, 'precision': 0.69, 'recall': 0.35, 'F1': 0.465, 'unlabeled': 16}
#  
#  {'total': 839, 'tp': 23, 'fp': 12, 'tn': 752, 'fn': 52, 'precision': 0.66, 'recall': 0.31, 'F1': 0.418, 'unlabeled': 18}
#  
#  {'total': 819, 'tp': 35, 'fp': 17, 'tn': 715, 'fn': 52, 'precision': 0.67, 'recall': 0.4, 'F1': 0.504, 'unlabeled': 8}
#  
#  {'total': 1873, 'tp': 87, 'fp': 55, 'tn': 1594, 'fn': 137, 'precision': 0.61, 'recall': 0.39, 'F1': 0.475, 'unlabeled': 4}
#  
#  {'total': 2157, 'tp': 289, 'fp': 139, 'tn': 1313, 'fn': 416, 'precision': 0.68, 'recall': 0.41, 'F1': 0.51, 'unlabeled': 555}

# %% [markdown]
# ## optimize table extraction parameters
#
# we can optimize for several goals:
#
# - Tp, precision, recall, F1

# %% tags=[]
def objective(trial: optuna.trial.Trial):
    # TODO: add a "pruner" which can automatically throw out unpromising trials where we don't find 
    # any tabl areas for a certain already labeled pdf...
    # we could do that by getting a subset of the files and
    # if we "should prune" https://optuna.readthedocs.io/en/stable/tutorial/10_key_features/003_efficient_optimization_algorithms.html
    # return the current value that we got
    # trial.should_prune()

    # as we are using a group trheshold of "10" a sensitivity of
    # "5" means 10/5 = 2 pixels. the higher the sensitiviy, the more precise the groups will be.
    # Or the less tolerance we have in orde to group boxes...
    hp = dict(
        es1 = trial.suggest_float(f"es1", 1.0, 20, log=True),  # edge sensitivity
        gs1 = trial.suggest_float(f"gs1", 1.0, 20, log=True),  # gap sensitivity
        es2 = trial.suggest_float(f"es2", 1.0, 20, log=True),
        gs2 = trial.suggest_float(f"gs2", 1.0, 20, log=True)
    )
    adp = [{
        "va": [hp['gs1'], hp['es1'], hp['es1'] / 2, hp['es1']],
        "ha": [hp['gs1'], hp['es1'], hp['es1'] / 2, hp['es1']]
    },
    {
        "va": [hp['gs2'], hp['es2'], hp['es2'] / 2, hp['es2']],
        "ha": [hp['gs2'], hp['es2'], hp['es2'] / 2, hp['es2']]
    }]
    ex_params = pydoxtools.extract_tables.TableExtractionParameters(
        area_detection_distance_func_params=adp,
        text_extraction_margin=10.0  # trial.suggest_float("text_extraction_margin", 0, 30)
    )

    tables = detect_tables(
        files, table_extraction_params=ex_params,
        max_filenum=50
    )

    tables_merged = merge_labels_by_closest_area(tables, tables_csv)

    stats = calc_stats(tables_merged)

    # we want to maximize recall for the areas, the precision can be
    # made bigger at the text-extraction stage..
    goal = stats["recall"]
    return goal

#previous_studies:
# table_area_X, 
study = optuna.create_study(f"sqlite:///{str(settings.TRAINING_DATA_DIR)}/table_X_studies.db", direction="maximize",
                            study_name="g_params_log_50", load_if_exists=True)

if not isnotebook():
    # make sure we have something that "works" in order to get going "faster" with out large parameter space
    #study.enqueue_trial({f"area_detection_params[{i}]": p for i, p in enumerate(initial_adp)})
    # current best area_detection_params
    #study.enqueue_trial({f"area_detection_params[{i}]": p for i, p in enumerate(best_adp)})
    study.optimize(objective, n_trials=100)

# %% tags=[]
[np.round(p, 2) for p in study.best_params.values()]
# [0.1, 3.1, 1.1,0.8,17.1, 9.1, 4.1,16.7]

# %%
study.best_params

# %%
# if not isnotebook():
# visualize parameter importances...
import optuna.visualization as ov
# import optuna.visualization.matplotlib as ov
if ov.is_available():
    figs = {
        # plot_intermediate_values
    # ov.plot_contour(study, params=study.best_params.keys()).write_html("contour.html")
        "param_importances.html": ov.plot_param_importances(study),
        "parallel_coordinate.html": ov.plot_parallel_coordinate(study, params=study.best_params.keys()),
        "optimization_history.html": ov.plot_optimization_history(study),
        "slice.html": ov.plot_slice(study),
        "edf.html": ov.plot_edf(study)
    }
    for key, fig in figs.items():
        fig.show()

# %%
# from IPython.core.display import display, HTML
# display(HTML(fig.to_html()))

# %% [markdown]
# ## check false negatives (not detected tables, but which were already labeled at some point...)
#
# we want to find interesting files to look at in "test_table_area_candidate_search.py" (why is it not working anymore?)

# %%
Fn.file_csv.unique()

# %% [markdown] tags=[]
# ## find tables that we haven't labeled yet within a certain tolerance

# %%
tol = 4 * 1.0  # maximum of XX pixels per edge on average in order to find a match

# %%
# selected_tables = tables_matched.query("match_dist>@tol & classification.isin(@good_area)")
selected_tables = tables_merged.query("match_dist>=@tol")
len(selected_tables)

# %%
selected_tables[['classification', 'match_dist']].head(3)

# %% [markdown] tags=[]
# ## label the new selection...

# %%
# label tables that we haven't labeled yet
print(len(selected_tables))
# selected_tables.head(5)

# %%
# plt.ioff()
def show_table(idx):
    table = selected_tables.loc[idx]
    print(f"idx: {idx}, \n{pathdb[table.file]} \npage: {table.page}")
    images = vda.cached_pdf2image(pathdb[table.file])
    p = table.pageobj
    margin = 100
    page_margins = table[pdf_utils.box_cols].values + [-margin, -margin, margin, margin]
    display(
        vda.plot_box_layers(
            box_layers=[
                # [p.df_ge_f[vda.box_cols].values, vda.LayerProps(alpha=0.1, color="red")],
                [p.df_le[vda.box_cols].values, vda.LayerProps(alpha=0.1, color="black")],
                [table[pdf_utils.box_cols].values[None, :],
                 vda.LayerProps(alpha=0.5, linewidth=.5, color="red", filled=False)],
                # [p.detect_table_area_candidates.values, vda.LayerProps(alpha=0.2, color="blue", filled=False)],
                # [p.table_candidate_boxes_df.values, vda.LayerProps(alpha=0.5, color="red", filled=False)],
                # [p.table_areas.values, vda.LayerProps(alpha=0.2, color="yellow")]
            ],
            image=images[int(table.page) - 1],
            image_box=p.page_bbox,
            bbox=page_margins, dpi=200
        )
        # pydoxtools.visual_document_analysis.plot_single_table_infos(
        #    page_bbox=table.page_bbox,
        #    image=images[int(table.page) - 1],
        #    table=table[pdf_utils.box_cols],
        #    dpi=100
        # )
    )
    display(table.table.style.set_table_attributes('style="font-size: 9px"'))
    plt.close()


# show_page_info(page)

from pigeonXT import annotate

annotations = annotate(
    selected_tables.index.to_list(),
    options=['all_good', 'area_correct','area_correct_checked', 'area_correct_checked_hf', 'wrong'],
    display_fn=show_table
)

# %%
raise ValueError("Stop Here!")  # this is simply a safety measure to not accidentally overwrite our labeled data...

# %%
save_tables = selected_tables.copy()
save_tables.loc[:, 'classification'] = annotations.label.to_list()

# %%
save_tables.loc[:, 'labeling_time'] = pd.Timestamp.now()

# %%
save_tables[int_cols] = save_tables[int_cols].astype(int)

# %%
save_tables[save_columns]

# %%
save_tables[save_columns].to_csv(
    table_data, mode='a', header=True, index=False)

# %%
