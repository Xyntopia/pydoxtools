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

# %% [markdown]
# # Extract tables and generate features which have a good correlaction with their labels
#
# the goal is to generate features which have the best possible correlation
# with the labels and thus can be used to recognize "good" and "bad" tables etc...
#
# in order to do this we need to have some very basic metrics for each table
# available which can then be used by a generator function to generate
# formulas to derive new features from those basic ones using a genetic algorithm approach
# where the corelation with the featurs is used as a correlation function.

# %% tags=[]
# %load_ext autoreload
# %autoreload 2

import itertools
import logging
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from IPython.display import display, HTML
from tqdm import tqdm

import pydoxtools.extract_tables
from pydoxtools import nlp_utils, labeling, func_generator as fg
from pydoxtools import pdf_utils, file_utils
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

files = file_utils.get_all_files_in_nested_subdirs(settings.PDFDIR, "*.pdf")
files = file_utils.get_all_files_in_nested_subdirs(settings.TRAINING_DATA_DIR / "pdfs/datasheet", "*.pdf")
# create a database so that we can lookup the actual paths fomr jsut the filenames
pathdb = {f.name: f for f in files}
len(files)

# %% [markdown] tags=[]
# ## load labeled tables from csv

# %%
# this should ALWAYS be predefined in order to keep the csv consistent
save_columns = sorted(['area_detection_method',
                       'classification', 'file', 'labeling_time',
                       'md5', 'page', 'x0', 'x1', 'y0', 'y1'])

# %% tags=[]
tables_csv = labeling.load_tables_from_csv()
print(tables_csv.columns)

# %% [markdown] tags=[]
# ## manually optimize and generate table statistics
#
# we need this, if we want to manually adapt the parameters in oder to label functions...

# %%
import warnings

warnings.filterwarnings('ignore')

if isnotebook() or getattr(sys, 'gettrace', None):
    tables = labeling.detect_tables(
        files, table_extraction_params=pydoxtools.extract_tables.TableExtractionParameters.reduced_params(),
        max_filenum=-1, cached=True
    )

# %% [markdown]
# ### match csv tables to detected tables

# %%
tables_merged = labeling.merge_table_classifications(tables, tables_csv)

# %% [markdown] tags=[]
# ## analysis

# %% [markdown]
# ### metrics correlation

# %%
print(tables_csv.classification.unique())

# %%
tables_labeled = tables_merged.query("_merge=='both'").dropna(axis=1).copy()
cols = [c for c in tables_labeled.columns if not c.endswith('_csv')]
tables_labeled = tables_labeled[cols]

# %%
classification_one_hot = pd.get_dummies(tables_merged.classification)
classification_one_hot.shape

# %%
# from pandas_profiling import ProfileReport
# profile = ProfileReport(tables, title="Pandas Profiling Report",  minimal=True)
# profile.to_widgets()
# library & dataset


# Basic correlogram
# sns.pairplot(tables[analysis_cols])
# plt.show()

# %%
classification_one_hot["supergood"] = classification_one_hot[['all_good', 'all_good-']].max(1)

# %%
# TODO:
features_one_hot_cols = ['area_detection_method']

# %%
# column definitions
metric_cols = [
    'x0', 'y0', 'x1', 'y1',
    'words_area_sum', 'word_line_num', 'word_count',
    'table_text_len', 'cells_num', 'cells_detected_num', 'hlines_num',
    'vlines_num', 'table_line_count', 'table_word_count',
    'graphic_elem_count', 'w', 'h', 'area', 'row_count', 'col_count',
    'empty_cells_sum', 'empty_cols_count', 'empty_rows_count',
    'cell_detected_size_sum', 'cells_span_num'
]

# %%
# numerical table columns that we forgot...
# tables_labeled.columns
# set(tables_labeled.select_dtypes(include=np.number).columns) - metric_cols

# %%
# table columns that are not taken into account as features right now ...
# set(tables_labeled.columns)-set(analysis_cols)-set(metric_cols_txt + metric_cols_area)

# %%
good_cols = ['supergood', 'all_good', 'all_good-']
bad_cols = ['wrong']
class_cols = good_cols + bad_cols

# %%
# sns.heatmap(tables_labeled[analysis_cols].corr(),  annot = True, fmt ='.1g', ax=ax, square=True)
analysis_table = tables_labeled.join(classification_one_hot)
# analysis_table.columns

# %%
f, ax = plt.subplots(figsize=(25, 5))
analysis_cols = metric_cols
sns.heatmap(analysis_table.corr()[metric_cols].loc[class_cols],
            annot=True, fmt='.2g', ax=ax, square=True)

# %% [markdown]
# ## generate more features

# %%
import operator
from sklearn.feature_selection import SelectKBest

max_iter_feats = 10

operators = [operator.truediv, operator.add, operator.mul, operator.sub]
# cols = metric_cols[:]# iterate over generated features again...
all_features = analysis_table[metric_cols].copy()
for i in range(2):
    corrdir = {}
    # add features which are already there ...
    # comb_feats={key: analysis_table[key] for key in metric_cols}
    comb_feats = {}
    # combine features

    kb = SelectKBest(k=min(100, all_features.shape[1])).fit(all_features.abs(), analysis_table['supergood'])
    all_features = all_features[set(kb.get_feature_names_out()).union(metric_cols)]
    print(all_features.shape)

    for v1, v2 in tqdm(list(itertools.permutations(all_features.columns, 2))):
        for func in operators:
            # comb_feat = analysis_table[v1]/(analysis_table[v2].abs()+1e-7)
            if func == operator.truediv:
                # prevent zero-division
                comb_feat = func(all_features[v1],
                                 (all_features[v2] + (np.sign(all_features[v2].values * 2 - 1)) * 1e-10))
            else:
                comb_feat = func(all_features[v1], (all_features[v2] + 1e-7))
            # comb_feat = analysis_table[[v1,v2]].max(axis=1)
            corr = analysis_table['supergood'].corr(comb_feat)
            key = "(" + v1 + fg.binary_func_dict[func] + v2 + ")"
            corrdir[key] = corr
            comb_feats[key] = comb_feat

    comb_feats = pd.DataFrame(comb_feats)
    comb_feats = comb_feats.drop(columns=all_features.columns.intersection(comb_feats.columns))
    all_features = all_features.join(comb_feats.dropna(axis=1))
    # all_features
    # remove low-variance features
    # vt = VarianceThreshold(threshold=(.9 * (1 - .9))).fit(all_features)
    # vt = VarianceThreshold(threshold=(.9 * (1 - .9))).fit(all_features)
    # only keep top correlation features
    # select best featues

    # best_feats = pd.Series(corrdir).dropna().abs().sort_values()
    # print(best_feats.tail(20).to_dict())
    # feat_select = best_feats.tail(max_iter_feats).index
    # all_features = all_features[feat_select.union(metric_cols)]
    # feat_select
    # all_features = all_features[all_features.columns[vt.get_support()]]

# %%
# all_features.columns

# %%
analysis_table.shape

# %%
best_feats = pd.Series(corrdir).dropna().sort_values()
print(best_feats.tail(20).to_dict())

feat_select = best_feats.tail(50).index.tolist()
# feat_select

# %% [markdown]
# ### all_features.isin([np.inf, -np.inf]).values.sum()

# %%
# try out functions...
# func = operator.truediv #mul sub add truediv
# v1,v2 = "col_count","empty_cols_count"
# func(analysis_table[v1], (analysis_table[v2] - (np.sign(analysis_table[v2].values*2-1))*1e-10))

# %%
all_features.isna().values.sum()

# %% [markdown] tags=[]
# ## sklearn classifier

# %%
import sklearn.tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.model_selection as model_selection

# %%
# all_estimators('classifier')

# %%
# comb_feats.dropna(axis=1).describe()

# %%
y = analysis_table['supergood']
X = all_features
X_train, X_test, y_train, y_test = model_selection.train_test_split(
    X[:], y.values, test_size=0.3, random_state=1)
clf = RandomForestClassifier(n_estimators=3,
                             max_depth=5, random_state=5,
                             class_weight={1: 1, 0: 1})
# clf = KNeighborsClassifier(n_neighbors=2)
clf = DecisionTreeClassifier(
    # max_depth=3,
    criterion="gini",  # "entropy" or "gini"
    # min_samples_leaf=0.05,
    random_state=20,
    # max_features=5,
    max_leaf_nodes=2,
    # class_weight="balanced"
    # class_weight={1: 1, 0: 0.001}
)
# clf = MLPClassifier(alpha=1, max_iter=10000)
clf.fit(X_train, y_train)
# clf.fit(X, y)
preds = clf.predict(X_test)

# preds_proba = clf.predict_proba(X_test)
# pd.DataFrame(preds_proba).hist(bins=100)

sklearn.metrics.accuracy_score(y_test, preds)
print(sklearn.metrics.classification_report(y_test, preds))

# %%
print(fg.tree_to_code(clf, feature_names=X.columns.tolist(), bool_output=True))


# %%
def tree(h, vlines_num, cells_span_num, words_area_sum, cells_num, cells_detected_num):
    if ((vlines_num / cells_detected_num) + (cells_span_num / cells_detected_num)) <= 1.2666667699813843:
        if ((words_area_sum / h) - (h / cells_num)) <= 11.474941492080688:
            return False  # classification scores: [[0.01 0.  ]]
        else:  # if ((words_area_sum/h)-(h/cells_num)) > 11.474941492080688
            return True  # classification scores: [[2.20e-02 3.59e+02]]
    else:  # if ((vlines_num/cells_detected_num)+(cells_span_num/cells_detected_num)) > 1.2666667699813843
        return False  # classification scores: [[0.022 0.   ]]


# %%
feature_names = ["h", "vlines_num", "cells_span_num", "words_area_sum", "cells_num", "cells_detected_num"]
preds = analysis_table[feature_names].apply(lambda x: tree(*x), axis=1)
print(sklearn.metrics.classification_report(analysis_table['supergood'], preds))

# %%

# %%
# import matplotlib.pyplot as plt  # make sure maplotlib is optional
# fig, ax = plt.subplots(dpi=200)
# sklearn.tree.plot_tree(clf, ax=ax, feature_names=X.columns.tolist(), class_names=True, fontsize=3);

# %%
# print(best_feats.tail(20).to_dict())
# print(best_feats.tail(50).index.tolist())

# %%
