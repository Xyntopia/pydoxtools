# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Label PDF table extractions

# %%
# %load_ext autoreload
# %autoreload 2

import concurrent.futures
import itertools
import logging
import sys

import numpy as np
import optuna
import pandas as pd
import warnings
import torch
from IPython.display import display, HTML
from matplotlib import pyplot as plt
from tqdm import tqdm

import pydoxtools.extract_tables
import pydoxtools.visual_document_analysis as vda
from pydoxtools import nlp_utils, labeling
from pydoxtools import pdf_utils, file_utils
from pydoxtools.labeling import find_closest_match_idx
from pydoxtools.settings import settings

logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
logging.getLogger('readability.readability').setLevel(logging.WARNING)

box_cols = pdf_utils.box_cols

tqdm.pandas()


def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))


pdf_utils._set_log_levels()
memory = settings.get_memory_cache()

nlp_utils.device, torch.cuda.is_available(), torch.__version__


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

files = file_utils.get_nested_paths(settings.PDFDIR, "*.pdf")
files = file_utils.get_nested_paths(settings.TRAINING_DATA_DIR / "pdfs/datasheet", "*.pdf")
# create a database so that we can lookup the actual paths fomr jsut the filenames
pathdb = {f.name: f for f in files}
len(files)

# %% [markdown]
# ## load labeled tables from csv

# %%
# this should ALWAYS be predefined in order to keep the csv consistent
save_columns = sorted(['area_detection_method',
                       'classification', 'file', 'labeling_time',
                       'md5', 'page', 'x0', 'x1', 'y0', 'y1'])

# %%
tables_csv = labeling.load_tables_from_csv()
print(tables_csv.columns)

# %% [markdown]
# ## manually optimize and generate table statistics
#
# we need this, if we want to manually adapt the parameters in oder to label functions...

# %%
import warnings
warnings.filterwarnings('ignore')

if isnotebook() or getattr(sys, 'gettrace', None):
    tables = labeling.detect_tables(
        files, table_extraction_params=pydoxtools.extract_tables.TableExtractionParameters.reduced_params(),
        max_filenum=-1
    )

# %% [markdown]
# ### match csv tables to detected tables

# %%
print(tables_csv.classification.unique())

# %%
tables_merged = labeling.merge_table_classifications(tables, tables_csv)

# %% [markdown]
# ## calculate statistics (Tp,Fp,Tn,Fn), unlabeled

# %%
good_areas = ['all_good', 'all_good-', 'area_correct', 'almost correct',
              'area_correct_with_header_footer',
              'area_correct_checked_hf', 'area_correct_checked']
good = ['all_good', 'all_good-']

# %%
Tp = tables_merged.query("_merge=='both' & classification.isin(@good)")
Fp = tables_merged.query("_merge=='both' & ~classification.isin(@good)")
Tn = tables_merged.query("_merge=='right_only' & ~classification.isin(@good_areas)")
Fn = tables_merged.query("_merge=='right_only' & classification.isin(@good_areas)")

# %% [markdown]
# #### eliminate a lot of false negatives:
#     
# if we have a "good" classification for a specific table, we can eliminate all matching tables which appear only on the right side up
# to a certain match_bbox_dist threshold. As we now know, that we correctly detected the table in the right area.

# %%
# match_indices relate to the detected tables *before* they got merged. Because of this
# we need to check the saved "index" column which contains the indices of the
# detected tables
Tpidx = Tp["index"].unique()
Fn = Fn.query("~(match_idx_bbox.isin(@Tpidx) & match_bbox_dist<200)")
Fn2 = Fn.query("classification.isin(@good)")

# %%
tables_csv.shape, tables.shape, tables_merged.shape

# %%
if isnotebook():
    tp, fp, tn, fn = len(Tp), len(Fp), len(Tn), len(Fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    F1 = 0 if (precision + recall) == 0 else 2 * (precision * recall) / (precision + recall)
    unlabeled = tables_merged.query("_merge=='left_only'")
    stats = dict(
        total=len(tables_merged),
        tp=tp, fp=fp,
        tn=tn, fn=fn, fn2=len(Fn2),
        precision=np.round(precision, 2),
        recall=np.round(recall, 2),
        F1=np.round(F1, 3),
        unlabeled=len(unlabeled),
    )
    print(stats)

# %% [markdown]
# TODO: add false negatives based on previous "good" classifications...

# %% [markdown]
# ### current state
#
# {'total': 2277, 'tp': 303, 'fp': 111, 'tn': 1051, 'fn': 48, 'fn2': 8, 'precision': 0.73, 'recall': 0.86, 'F1': 0.792, 'unlabeled': 684}
#
# {'total': 2476, 'tp': 483, 'fp': 58, 'tn': 1172, 'fn': 24, 'fn2': 3, 'precision': 0.89, 'recall': 0.95, 'F1': 0.922, 'unlabeled': 481}
#
# {'total': 2476, 'tp': 512, 'fp': 79, 'tn': 1172, 'fn': 24, 'fn2': 3, 'precision': 0.87, 'recall': 0.96, 'F1': 0.909, 'unlabeled': 431}
#
# {'total': 2394, 'tp': 510, 'fp': 32, 'tn': 1283, 'fn': 28, 'fn2': 5, 'precision': 0.94, 'recall': 0.95, 'F1': 0.944, 'unlabeled': 283}

# %% [markdown]
# ### diagnostics

# %% [markdown]
# TODO: analyze False positives with area-correct

# %% [markdown]
# False negatives (the ones we SHOULD detct, but do not ...)

# %%
Fn[Fn.match_bbox_dist < 150].match_bbox_dist.hist(bins=50)

# %%
Fn_files = (settings.TRAINING_DATA_DIR / "pdfs/datasheet" / Fn.file_csv).value_counts()
pretty_print(Fn_files.to_frame().head(10))
Fn_files.shape
# Fp_files.shape

# %% [markdown]
# which tables in a specific file were classified as wrong? and why?
# - if they have small *match_bbox_dist* and we have a "good" classification based on md5 on the matching index,
#   we can eliminate these tables from false negatives... So basically, there shouldn't be any close bbox 
#   matches here...

# %%
tables_merged.columns

# %%
file = "2239470.28.pdf"
Fn.query("file_csv==@file")[[
                                "classification", "match_idx_bbox", "match_bbox_dist", "match_idx", "_merge",
                                "page_csv",
                                "labeling_time", "index"
                            ] + [b + "_csv" for b in box_cols]]

# %%
tables_merged.query("file==@file")[[
                                       '_merge', 'classification', "match_idx_md5", "match_idx_bbox", "match_bbox_dist",
                                       "labeling_time", "index"] + box_cols]

# %% [markdown]
# Files that were labeled but do not appear in our dataset:

# %%
t = tables_csv.query("~file.isin(@tables.file.unique())").file.drop_duplicates()
pretty_print((settings.TRAINING_DATA_DIR / "pdfs/datasheet" / t).to_frame())

# %% [markdown]
# ## optimize table extraction parameters
#
# we can optimize for several goals:
#
# - Tp, precision, recall, F1

# %%
optimize = False


# %%
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
        es1=trial.suggest_float(f"es1", 1.0, 20, log=True),  # edge sensitivity
        gs1=trial.suggest_float(f"gs1", 1.0, 20, log=True),  # gap sensitivity
        es2=trial.suggest_float(f"es2", 1.0, 20, log=True),
        gs2=trial.suggest_float(f"gs2", 1.0, 20, log=True)
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


# previous_studies:
# table_area_X, 
study = optuna.create_study(f"sqlite:///{str(settings.TRAINING_DATA_DIR)}/table_X_studies.db", direction="maximize",
                            study_name="g_md5_match_params_log_50", load_if_exists=True)

if not isnotebook():
    # make sure we have something that "works" in order to get going "faster" with out large parameter space
    # study.enqueue_trial({f"area_detection_params[{i}]": p for i, p in enumerate(initial_adp)})
    # current best area_detection_params
    # study.enqueue_trial({f"area_detection_params[{i}]": p for i, p in enumerate(best_adp)})
    study.optimize(objective, n_trials=100)

# %%
if optimize:
    print([np.round(p, 2) for p in study.best_params.values()])
    # [0.1, 3.1, 1.1,0.8,17.1, 9.1, 4.1,16.7]

    print(study.best_params)

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
# we want to find interesting files to look at in "test_table_area_candidate_search.py" (a.k.a. why is the detection not working anymore?)

# %%
pretty_print(str(settings.TRAINING_DATA_DIR / "pdfs/datasheet") + "/" + pd.DataFrame(Fn2.file_csv.unique()))

# %% [markdown]
# ## Calculate some automatic labeling sets

# %% [markdown]
# ### find tables that we haven't labeled yet within a certain tolerance

# %%
# selected_tables = tables_matched.query("match_dist>@tol & classification.isin(@good_area)")
selected_tables = tables_merged.query("_merge=='right_only'")[:50]
len(selected_tables)

# %% [markdown]
# ### find previously labeled tables by area in order to automatically label them

# %%
tables_csv.classification.unique()

# %%
tables_merged.columns

# %%
tables_merged.query("match_bbox_dist>50").shape

# %%
tables_merged.query("match_bbox_dist<=1 & classification.isin(@good)").shape

# %% [markdown] jp-MarkdownHeadingCollapsed=true
# select all unlabeled closest area-matches to false negatives

# %%
selected_tables = unlabeled.merge(Fn, how="inner", right_on="match_idx_bbox", left_index=True, suffixes=(None, "_csv"))

# %%
# we can change this line to different things in roder to get the subsets...
# selected_tables = tables_merged_area.query("match_dist<=10")
# selected_tables = tables_merged_area.query("classification.isna()")
selected_tables = tables_merged.query("_merge=='right_only' & classification.isin(@good)")
# selected_tables = tables_merged_area.query("match_dist<=1 & classification.isin(@good)")
# selected_tables = tables_merged_area.query("match_dist>20")
# selected_tables=Fn2 # doesn't wrk, because they werent detected as table objects
# selected_tables=unlabeled.query("file==@file")
selected_tables = unlabeled
selected_tables=Fp
selected_tables.shape, tables_merged.shape

# %% [markdown]
# ## label the new selection...

# %%
max_labels = 50
# label tables that we haven't labeled yet
print(len(selected_tables))
# selected_tables.head(5)
#selected_tables = selected_tables.sample(50)

# %%
# plt.ioff()
def show_table(idx):
    table = selected_tables.loc[idx]
    print(f"idx: {idx}, \n{pathdb[table.file]} \npage: {table.page}")
    print(f"previous classification: {table.classification}")
    images = vda.cached_pdf2image(pathdb[table.file])
    p = table.pageobj
    t = table.tableobj
    margin = 20
    page_margins = table[pdf_utils.box_cols].values + [-margin, -margin, margin, margin]
    display(
        vda.plot_box_layers(
            box_layers=[
                # [p.df_ge_f[vda.box_cols].values, vda.LayerProps(alpha=0.1, color="red")],
                # [p.df_le[vda.box_cols].values, vda.LayerProps(alpha=0.1, color="black")],
                [t.df_words[vda.box_cols].values, vda.LayerProps(alpha=0.1, color="black")],
                [table[pdf_utils.box_cols].values[None, :],
                 vda.LayerProps(alpha=0.5, linewidth=.5, color="red", filled=False)],
                # [p.detect_table_area_candidates.values, vda.LayerProps(alpha=0.2, color="blue", filled=False)],
                # [p.table_candidate_boxes_df.values, vda.LayerProps(alpha=0.5, color="red", filled=False)],
                # [p.table_areas.values, vda.LayerProps(alpha=0.2, color="yellow")]
            ],
            image=images[int(table.page) - 1],
            image_box=p.page_bbox,
            bbox=page_margins, dpi=150
        )
        # pydoxtools.visual_document_analysis.plot_single_table_infos(
        #    page_bbox=table.page_bbox,
        #    image=images[int(table.page) - 1],
        #    table=table[pdf_utils.box_cols],
        #    dpi=100
        # )
    )
    pretty_print(table.table)
    plt.close()


# show_page_info(page)

from pigeonXT import annotate

annotations = annotate(
    selected_tables.index.to_list(),
    options=['all_good', 'all_good-', 'area_correct', 'area_correct_checked', 'area_correct_checked_hf', 'wrong'],
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
save_tables[save_columns].query("classification!=''")

# %%
save_tables[save_columns].query("classification!=''").to_csv(
    table_data, mode='a', header=True, index=False)

# %%
