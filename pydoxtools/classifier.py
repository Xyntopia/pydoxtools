#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 22:01:10 2020
TODO write file description
"""

import abc
import concurrent.futures
import datetime
import functools
import logging
import multiprocessing as mp
import pathlib
import pickle
import random
import re
import string
import typing
import urllib
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import pytorch_lightning
import torch
import torch.nn.functional as F
import torchmetrics
import tqdm
from transformers import AutoTokenizer, AutoModel

from pydoxtools import html_utils, file_utils, pdf_utils, list_utils
from pydoxtools.settings import settings

logger = logging.getLogger(__name__)
memory = settings.get_memory_cache()
tqdm.tqdm.pandas()

import sklearn
import sklearn.model_selection


class nlp_functions:
    def init_embeddings(self):
        """
        This function can be used to pre-initialize the embeddings using
        for example BERT embeddings.

        We freeze the embeddings used here so that we can generalize the vectorization
        for many different documents and tasks. this way we can generate the vectorizations
        once and then store them in a database and reuse them for all kinds of different tasks.

        """
        model = AutoModel.from_pretrained(self.model_name)

        self.embedding = torch.nn.Embedding.from_pretrained(
            model.embeddings.word_embeddings.weight,
            freeze=True, padding_idx=None, max_norm=None,
            norm_type=2.0, scale_grad_by_freq=False, sparse=False
        )
        return self


class string_vectorizer(torch.nn.Module):
    """
    This module takes a webpage and finds an embedding using
    a pre-generated look-up table.56
    """

    def __init__(self):
        super(string_vectorizer, self).__init__()
        self.model_name = 'distilbert-base-multilingual-cased'  # name of model used for initialization
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        embedding_num, embedding_dim = 119547, 768  # size of multilingual BERT vocabulary
        self.embedding = torch.nn.Embedding(embedding_num, embedding_dim)

    def init_embeddings(self):
        """
        This function can be used to pre-initialize the embeddings using
        for example BERT embeddings.

        We freeze the embeddings used here so that we can generalize the vectorization
        for many different documents and tasks. this way we can generate the vectorizations
        once and then store them in a database and reuse them for all kinds of different tasks.

        """
        model = AutoModel.from_pretrained(self.model_name)

        self.embedding = torch.nn.Embedding.from_pretrained(
            model.embeddings.word_embeddings.weight,
            freeze=True, padding_idx=None, max_norm=None,
            norm_type=2.0, scale_grad_by_freq=False, sparse=False
        )
        return self

    def forward(self, strlist):
        # TODO: we might be able to us "diskcache" here in order to cache some layers!!!
        #       for example the tokenized ids
        ids = torch.tensor(self.tokenizer(strlist).input_ids)
        x = self.embedding(ids)
        return torch.mean(x, -2)


def html_clean(raw_html, cut_len=50000):
    try:
        clean = memory.cache(html_utils.get_pure_html_text)
        return clean(raw_html)[:cut_len]
    except:
        logger.info(f"cleaning did not work")
        return "ERROR"


@memory.cache(ignore=["model"])
def cached_str_vectorize(model, raw_string):
    return model(raw_string)


class classifier_functions:
    @abc.abstractmethod
    def generate_features(self, X):
        return X

    def forward_text(self, X):
        x = self.generate_features(X)
        return self(x)

    def predict_proba_from_featurevec(self, X):
        x = torch.tensor(X).float()
        x = self.classification(x)
        x = F.softmax(x, dim=1)
        return x

    def predict_proba(self, X):
        x = self.forward(X)
        return F.softmax(x, dim=1)

    def predict(self, X):
        x = self.forward(X)
        x = torch.argmax(x, 1)
        return [self.classmap_[i.item()] for i in x]

    def predict_proba_feat(self, X):
        x = self.forward_text(X)
        return F.softmax(x, dim=1)

    def predict_feat(self, X):
        x = self.forward_text(X)
        x = torch.argmax(x, 1)
        return [self.classmap_[i.item()] for i in x]

    def predict_info(self, X):
        x = self.forward_text(X)
        probs = torch.max(F.softmax(x, dim=-1), dim=-1).values
        x = torch.argmax(x, -1)
        return {
            "classes": [self.classmap_[i.item()] for i in x],
            "probabilities": probs
        }


class lightning_training_procedures(pytorch_lightning.LightningModule):
    def __init__(self):
        super(lightning_training_procedures, self).__init__()
        self.histograms = True
        self.class_weight = torch.tensor([1., 1.])

    def add_standard_metrics(self, num_classes, hist_params=None):
        # TODO: get rid of this function in all child classes and somehow
        #       calculate "num_classes" automatically during initialization
        # define sensors for training/testing
        if not hist_params:
            self.hist_params = ['linear.weight', 'linear.bias']
        else:
            self.hist_params = hist_params
        threshold = 0.5
        self.metrics = torch.nn.ModuleDict({
            'accuracy': torchmetrics.Accuracy(threshold, num_classes),
            'f1': torchmetrics.F1(num_classes, threshold=threshold),
            'recall': torchmetrics.Recall(num_classes, threshold=threshold),
            'precision': torchmetrics.Precision(num_classes, threshold=threshold)
        })

    def training_step(self, batch, batch_idx):
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y_train_pred = self(x)
        # this function handles th multiple categorical outputs from
        # y_train_pred and numerical categories from y automatically...
        loss = F.cross_entropy(
            y_train_pred, y,
            weight=self.class_weight.to(device=self.device)
        )
        # Logging to TensorBoard by default
        # self.log('train_loss', loss)
        self.log(
            'train_loss', loss, on_step=True,
            on_epoch=True, prog_bar=True, logger=True,
            batch_size=len(batch)
        )
        return loss

    def custom_histogram_adder(self):
        if self.histograms:
            for name, params in self.named_parameters():
                if name in self.hist_params:
                    self.logger.experiment.add_histogram(name, params, self.current_epoch)

    def training_epoch_end(self, training_step_outputs):
        self.custom_histogram_adder()

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self(x)
        target = y
        return preds, target

    def test_epoch_end(self, test_step_outputs):
        pred, target = list(zip(*test_step_outputs))
        pred, target = torch.cat(pred), torch.cat(target)
        for metric_name in self.metrics:
            self.metrics[metric_name](pred, target)
            self.log(metric_name, self.metrics[metric_name])

        classes = [v for k, v in self.classmap_.items()]
        est = pd.DataFrame(pred.cpu().numpy(), columns=classes)
        est['target'] = target.cpu().numpy()
        est['max'] = est.loc[:, classes].max(axis=1)  # get the calculated certainty
        est['maxidx'] = est.loc[:, classes].idxmax(axis=1)  # get the label
        est['target_label'] = est['target'].map(self.classmap_)

        # Model Accuracy
        predicted, test = est[['maxidx', 'target_label']].values.T
        accuracy = sklearn.metrics.accuracy_score(test, predicted)

        return {
            # "tests": est,
            "classification_report": sklearn.metrics.classification_report(test, predicted)
        }

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def validation_epoch_end(self, val_step_outputs):
        pred, target = list(zip(*val_step_outputs))
        pred, target = torch.cat(pred), torch.cat(target)
        for metric_name in self.metrics:
            self.metrics[metric_name](pred, target)
            self.log(metric_name, self.metrics[metric_name])

    def configure_optimizers(self):
        weight_decay = 0.0
        learning_rate = 0.001
        optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate,
            weight_decay=weight_decay
        )
        return optimizer

class txt_block_classifier(
    nlp_functions,
    lightning_training_procedures,
    classifier_functions,
    pytorch_lightning.LightningModule
):
    """
    This model takes a string and classifies it.
    """

    def __init__(self, classmap):
        super(txt_block_classifier, self).__init__()
        self.save_hyperparameters()  # make sure we have
        self.classmap_ = classmap
        self.classmapinv_ = {v: k for k, v in classmap.items()}
        self.classes_ = list(classmap.values())
        num_classes = len(classmap)
        self.class_weight = torch.tensor([0.75, 1.4])
        super(txt_block_classifier, self).__init__()

        # TODO: get rid of model dependency... only use the vocabulary for the tokenizer...
        self.model_name = 'distilbert-base-multilingual-cased'  # name of model used for initialization
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        embedding_num = 119547  # size of multilingual BERT vocabulary
        embedding_dim = 768  # size of bert token embeddings

        # TODO: the 2D paramters of the AvgPool would be a really nice parameter
        #       for hyperparameter optimization
        # TODO: checkout if it might be a better idea to use transformers for this task?
        # TODO: finetune a hugginface transformer for this task?
        # TODO: use some hyperparameter optimization for all the stuff below...
        # self.cv1 = torch.nn.Conv1d(
        #    in_channels=1, out_channels=10,
        #    kernel_size=768, stride=1) # reduce size of word vectors
        reduced_embeddings_dim = 16  # reduced feature size
        # TODO: the following is only needed if we use pretrained embeddings
        # self.l1 = torch.nn.Linear(in_features=embedding_dim, out_features=reduced_embeddings_dim)
        self.embedding = torch.nn.Embedding(embedding_num, reduced_embeddings_dim)
        self.dropout1 = torch.nn.Dropout(p=0.5)
        token_seq_length1 = 5  # what length of a work do we assume in terms of tokens?
        seq_features1 = 40  # how many filters should we run for the analysis = num of generated features?
        self.cv1 = torch.nn.Conv2d(
            in_channels=1,  # only one layer of embeddings
            out_channels=seq_features1,  # num of encoded features/word
            kernel_size=(
                token_seq_length1,
                reduced_embeddings_dim  # we want to put all features of the embedded word vector through the filter
            ),
            # the second dimension of stride has no effect, as our filter has the same size
            # as the vector anyways and we can leave it at 1
            stride=(token_seq_length1 // 2, 1)
        )
        token_seq_length2 = 40  # how many tokens in a row do we want to analyze?
        seq_features2 = 100  # how many filters should we run for the analysis?
        self.cv2 = torch.nn.Conv2d(
            in_channels=1,  # only one layer of embeddings
            out_channels=seq_features2,  # num of encoded features/word
            kernel_size=(
                # we have to switch around features and seq length, as cv1 puts
                # the kernel features before the new sequence length
                seq_features1,
                token_seq_length2
            ),
            # this time we have to switch around the stride as well
            stride=(1, token_seq_length2 // 2)
        )
        self.dropout2 = torch.nn.Dropout(p=0.5)
        # afterwards we do a max pooling and feed the input into a linear layer
        # we add addtional features here...
        # right now: 1: length of string + average embeddings vector for entire string
        meta_features = 1 + reduced_embeddings_dim
        self.linear = torch.nn.Linear(
            in_features=seq_features2 + meta_features, out_features=num_classes
        )

        # and add metrics
        self.add_standard_metrics(num_classes, hist_params=[
            "linear.weight", "linear.bias",
            "cv2.weight", "cv2.bias",
            "cv1.weight", "cv1.bias"
        ])

    def forward(self, strlist):
        # TODO: we might be able to us "diskcache" here in order to cache some layers!!!
        #       for example the tokenized ids
        lengths = torch.tensor([len(s) for s in strlist]).unsqueeze(1).to(device=self.device)
        # TODO: add optional str.strip here... (deactivated during training to improve speed)
        ids = self.tokenizer(
            list(strlist),  # convert trlist into an actual list (e.g. if it is a pandas dataframe)
            padding='max_length', truncation=True,
            max_length=500,  # approx. half an A4 page of text
            return_tensors='pt',
            return_length=True  # use this as another feature
        ).input_ids.to(device=self.device)
        # x = torch.Tensor(ids.shape[0], ids.shape[1])
        # ids = torch.tensor(ids)
        x = self.embedding(ids)
        embeddings_mean = x.mean(axis=1)
        x = self.dropout1(x)
        # x = self.l1(x) # if we use "frozen" embeddings we want to transform them into a more usable format...
        # "unsqueeze" to add the channl dimensions and then "squeeze" out the channel at the end...
        x = self.cv1(x.unsqueeze(1)).squeeze(3)
        x = self.dropout2(x)
        x = self.cv2(x.unsqueeze(1)).squeeze(2)
        x = torch.max(x, dim=2).values
        # combine features
        x = torch.cat([x, lengths, embeddings_mean], 1)
        x = self.linear(x.flatten(1))
        return x

# TODO: merge with txtblockclassifier
class urlClassifier(
    lightning_training_procedures,
    classifier_functions
):
    """
    this classifier takes a URL and classifies it in categories. For example if the word
    "product" appears in the URL their is a high chance of it referring to
    a webpage describing a product.

    TODO: right now the model can only cope with batches of size: 1, it would
          be preferrable to have larger batches.

    TODO: make urlClassifier a descendent of txtClassifier
    """

    def __init__(self, classes):
        super(urlClassifier, self).__init__()
        self.save_hyperparameters()
        self.classes_ = list(classes)
        self.classmap_ = dict(enumerate(classes))
        num_classes = len(classes)

        # define network layers
        self.string_vectorizer = string_vectorizer()
        outvec = self.string_vectorizer.embedding.weight.shape[1]
        self.dropout = torch.nn.Dropout(p=0.2)
        self.out = torch.nn.Linear(outvec, len(classes))

        self.add_standard_metrics(num_classes)

    def forward(self, urls):
        x = self.string_vectorizer(urls)
        # x = torch.max(torch.tensor(x), 0).values
        x = self.dropout(x)
        x = self.out(x)
        return x

# TODO: merge with pdfClassifier into a "documentClassifier"
class pageClassifier(
    lightning_training_procedures,
    classifier_functions
):
    """
    this classifier takes a webpage + url and classifies it in categories.

    The model can be initialized with embeddings from multilingual BERT.
    If we unfreeze them we could theoretically also finetune them
    on our data. Right now we don't do this.

    TODO: make the clasifier itself check if it got the right "vectorization"
          for classification (in the case of pre-generated vectorizations from
          a database)
    """
    __version__ = "CCXv2002007"

    def __init__(self, classmap, scaling):
        super(pageClassifier, self).__init__()
        self.save_hyperparameters()
        self.classmap_ = classmap
        self.classmapinv_ = {v: k for k, v in classmap.items()}
        self.classes_ = list(classmap.values())
        self.scaling = torch.tensor(scaling)
        num_classes = len(classmap)
        self.testing = False
        feature_info = dict(
            url_featuresize=768,
            html_featuresize=768,
            metainfo_featuresize=5,
        )
        featuresize = sum(v for k, v in feature_info.items())

        self.string_vectorizer = string_vectorizer()
        self.dropout = torch.nn.Dropout(p=0.2)
        self.linear = torch.nn.Linear(featuresize, num_classes)

        self.add_standard_metrics(num_classes)

    def forward(self, X):
        """
        takes only a single page right now if not in
        training mode
        """
        # if self.training or self.testing:
        #    x = self.dropout(X)
        # else:
        #    x = self.generate_features(X)
        x = self.dropout(X)
        x = self.classification(x)
        return x

    def classification(self, x):
        return self.linear(x)

    # @functools.lru_cache(maxsize=10)
    def generate_features(self, pagelist: List[Tuple[str, str]]) -> torch.TensorType:
        """
        We can use lru_cache here because the model will always the generate the same
        feature vector for a specific input.

        # split the vector of a page into several sections:

        # 1 section which takes the header into account
        # another one for other pars of the page...
        # another one for another part?  like iframe? sidebar? links?
        # another vector just for titles?
        """
        x = []
        for page in pagelist:
            url, html = page
            meta = gen_meta_info(url, html)
            meta = torch.log(meta + 1.0) / self.scaling  # normalize metadata
            url_vec = self.string_vectorizer(url)
            clean_html = html_clean(html)
            html_vec = self.string_vectorizer(clean_html)
            x.append(torch.hstack((meta, url_vec, html_vec)))
        return torch.stack(x)


class FailedVectorization(Exception):
    pass


class pdfClassifier(
    lightning_training_procedures,
    classifier_functions
):
    """
    this classifier takes a piece of text and classifies it...
    """
    __version__ = "txt20200807"

    def __init__(self, classmap):
        super(pdfClassifier, self).__init__()
        self.save_hyperparameters()  # save classmap
        self.cached = False
        self.classmap_ = classmap
        self.classmapinv_ = {v: k for k, v in classmap.items()}
        self.classes_ = list(classmap.values())
        num_classes = len(classmap)
        self.testing = False

        self.string_vectorizer = string_vectorizer()
        self.dropout = torch.nn.Dropout(p=0.2)
        featuresize = 768 + 768 + 4  # for BERT vectorizations
        self.linear = torch.nn.Linear(featuresize, num_classes)

        self.add_standard_metrics(num_classes)

    def forward(self, X) -> torch.TensorType:
        """
        takes only a single page right now if not in
        training mode
        """
        # if self.training or self.testing:
        #    x = self.dropout(X)
        # else:
        #    x = self.generate_features(X)
        x = self.dropout(X)
        x = self.classification(x)
        return x

    def classification(self, x) -> torch.TensorType:
        # the classification neural network ...
        return self.linear(x)

    def generate_features(self, pdf_list: List[typing.Union[pathlib.Path, typing.IO]]) -> torch.Tensor:
        """
        TODO: add some metadata to the equation:
                - number of pages (0 := unknown)
                - number of words, lines, average length of words
                - number of units
                - statistics: numbercount vs. wordcount
                - number of figures
                - number of tables
                - filename

        TODO: generate vectors for each page ...
        """
        max_pages = 5
        x = []
        pdftxtfunc = pdf_utils.get_pdf_text_safe_cached if self.cached else pdf_utils.get_pdf_text_safe
        pdfmeta = pdf_utils.get_meta_infos_safe_cached if self.cached else pdf_utils.get_meta_infos_safe
        for pdf in pdf_list:
            try:
                filename = pdf.name
                filename_vec = self.string_vectorizer(filename)
                txt = pdftxtfunc(pdf, maxpages=max_pages)
                txt_vec = self.string_vectorizer(txt)
                # meta = gen_meta_info(url, html)
                # meta = torch.log(meta + 1.0) / self.scaling  # normalize metadata
                metadata = pdfmeta(pdf)
                page_num = metadata['pagenum']
                txt_split = txt.split()
                word_num = sum(1 for s in txt_split if s.isalpha())
                digit_num = sum(1 for s in txt_split if s.isdigit())
                extracted_page_num = min(page_num, max_pages)
                meta_vec = torch.Tensor([
                    page_num / 50.0,
                    len(txt_split) / extracted_page_num,
                    word_num / extracted_page_num,
                    digit_num / extracted_page_num,
                ])
                x.append(torch.hstack((filename_vec, txt_vec, meta_vec)))
            except:
                logger.exception("failed vectorization...")
                raise FailedVectorization(f"converting {pdf} did not work")

        return torch.stack(x)


@functools.lru_cache()
def load_classifier(name):
    logger.info(f"loading classifier: {name}")
    if name == "text_block":
        net = txt_block_classifier.load_from_checkpoint(settings.CLASSIFIER_STORE(name))
    elif name == "url":
        net = urlClassifier.load_from_checkpoint(settings.CLASSIFIER_STORE(name))
    elif name == "page":
        net = pageClassifier.load_from_checkpoint(settings.CLASSIFIER_STORE(name))
    elif name == "pdf":
        net = pdfClassifier.load_from_checkpoint(settings.CLASSIFIER_STORE(name))

    net.freeze()
    return net


@functools.lru_cache()
def gen_meta_info(url, htmlstr) -> torch.tensor:
    """
    Generates some meta-info mainly based
    on the structure of a page:

    This is a stand-alone function in order to be able to cache it...

    TODO: generate more metadata:
    - has pdf link
    - text density
    - body length
    - header length
    - count tag-numbers https://stackoverflow.com/questions/6483851/is-there-an-elegant-way-to-count-tag-elements-in-a-xml-file-using-lxml-in-python
    """

    try:
        clean_html_str = html_clean(htmlstr, 2000000)
        urlobj = urllib.parse.urlsplit(url)
        x = [
            len(url),
            len(urlobj.path.split('/')),
            len(urlobj.query),
            len(htmlstr),
            len(clean_html_str)
        ]
    except ValueError:
        logger.info(f"Could not extract metainfo from {url}")
        x = [0, 0, 0, 0, 0]
    return torch.tensor(x)


gen_meta_info_cached = memory.cache(gen_meta_info)


@functools.lru_cache()
def load_labeled_webpages():
    trainingfiles = file_utils.get_all_files_in_nested_subdirs(settings.TRAINING_DATA_DIR, "trainingitem*.parquet")

    df = pd.read_parquet(trainingfiles[0])
    df['file'] = trainingfiles[0]
    for tf in trainingfiles[1:]:
        df2 = pd.read_parquet(tf)
        df2['file'] = tf
        df = df.append(df2, ignore_index=True)

    pagetypes = df['page_type'].unique().tolist()

    # convert a list of types into "unknown"
    if False:
        typemap = {
            'mainpage': 'unknown',
            'product_list': 'unknown',
            'productlist': 'unknown',
            'blog': 'unknown',
            'software': 'component',
            'company': 'unknown'}
        for k, v in typemap.items():
            df.loc[df.page_type == k, 'page_type'] = v
    else:
        allowed_types = ['unknown', 'error', 'component']
        for pt in pagetypes:
            if not pt in allowed_types:
                df.loc[df.page_type == pt, 'page_type'] = 'unknown'

    df.page_type = df.page_type.astype("category")
    return df


@functools.lru_cache()
def load_labeled_pdf_files(label_subset=None) -> pd.DataFrame:
    """
    This function loads the pdfs from the training data and
    labels them according to the folder they were found in...

    TODO: use the label_subset
    """

    # TODO: remove settings.trainingdir and movethis to "analysis" or somewhere else...
    df = file_utils.get_all_files_in_nested_subdirs(settings.TRAINING_DATA_DIR, '*.pdf')
    df = pd.DataFrame([Path(f) for f in df], columns=["path"])

    df['name'] = df.path.apply(lambda x: x.name)
    # pdf_files['parent']
    df['class'] = df.path.apply(lambda x: x.parent.name)
    if label_subset:
        allowed_types = label_subset
    else:
        allowed_types = ['unknown', 'datasheet', 'certificats', 'datasheetX']

    # throw out unlabeled datasheets:
    df = df.loc[~df['class'].isin(['unlabeled', 'downloaded'])]

    for pdf_type in df['class'].unique():
        if not (pdf_type in allowed_types):
            df.loc[df['class'] == pdf_type, 'class'] = 'unknown'

    df.loc[df['class'] == 'datasheetX', 'class'] = 'datasheet'

    df['class'] = df['class'].astype("category")
    return df


def get_pdf_txt_box(fo):
    try:
        return pdf_utils.get_pdf_text_safe_cached(fo, boxes=True)
    except:
        logger.exception(f"we have a problem with {fo}")
        return []


@memory.cache
def get_pdf_text_boxes(max_filenum: int = None, extended=False) -> pd.DataFrame:
    # get all pdf files in subdirectories
    files = file_utils.get_all_files_in_nested_subdirs(settings.TRAINING_DATA_DIR / "pdfs", "*.pdf")
    if extended:  # add more txtblock datasets
        files += file_utils.get_all_files_in_nested_subdirs(settings.DATADIR / "pdfs", "*.pdf")

    logger.info("now loading pdfs and generating text blocks...")

    txtboxes = set()
    with concurrent.futures.ProcessPoolExecutor(mp_context=mp.get_context('spawn')) as executor:
        for fn, txtbox_list in zip(files, tqdm.tqdm(executor.map(
                # pdf_utils.get_pdf_text_safe_cached, files[:max_filenum]
                get_pdf_txt_box,
                files[:max_filenum]
        ))):
            txtboxes.update((tb, fn) for tb in txtbox_list)

    # no cuncurrency
    # for pdf_file in tqdm.tqdm(files[:max_filenum]):
    #    boxes = pdf_utils.get_pdf_text_safe_cached(pdf_file, boxes=True)
    #    txtboxes.extend(boxes)
    txtboxes = pd.DataFrame(txtboxes, columns=["txt", "filename"])
    return txtboxes


def rand_chars(char_dict: dict) -> typing.Callable[[], str]:
    """create a random function which randomly selects characters from
    a list. function Argument is a dictionary with weights on how often
    characters should be chosen:

    separators = {
        " ": 10, ",": 2, "\n": 4, ";": 2, " |": 1,
        ":": 1, "/": 1
    }
    rand_seps = rand_chars(separators)

    would result in " " getting chosen ten times as often as "/".
    """

    def rand_func():
        return random.choices(list(char_dict.keys()),
                              weights=list(char_dict.values()))[0]

    return rand_func


# TODO: move all the following functions into a separate
#       "data preparation" file and only load the generated data here...
@memory.cache
def generate_random_textblocks(filename, iterations=3):
    """load a list of random textpieces from a parquet file
    and generate augmented textblocks from it..."""

    def weibull_ints():
        k = 1.4  # scale
        l = 8  # shape
        return int(random.weibullvariate(l, k))

    separators = {
        " ": 10, ",": 2, "\n": 4, ";": 2, " |": 1,
        ":": 1, "/": 1
    }
    rand_seps = rand_chars(separators)

    txtboxes = set()
    for n in range(iterations):
        # as we are doing a random process we can run this in multiple iterations
        # producing completly different textblocks
        for txt in tqdm.tqdm(pd.read_parquet(filename).txt):
            randtxt = [rand_seps().join(r) for r in
                       list_utils.random_chunks(txt.split(), weibull_ints)]
            txtboxes.update(randtxt)

    return txtboxes


@memory.cache
def fake_company_address_collection(size: int, fake_langs: List[str]) -> List[str]:
    from faker import Faker
    fake = Faker(fake_langs)
    rc = random.choice
    r = random.random

    def gen_company_address():
        # TODO: international addresses
        # TODO: more variations
        urlreplace = re.compile(r'[^a-zA-Z0-9]+')
        company: str = fake.company()
        phone = rc(["", "Phone ", "Phone: "]) + fake.phone_number()
        fax = rc(["", "fax ", "fax: "]) + fake.phone_number()
        if r() < 0.3:
            fax = fax.upper()
            phone = phone.upper()
        parts = dict(
            name=company if r() > 0.4 else company.upper(),
            address=fake.address(),
            phone=phone,
            fax=fax,
            www=rc(["http://", ""]) + rc(
                ["www.", ""]) + f"{urlreplace.sub(rc([r'-', '']), company)}.{fake.tld()}"
        )
        # leave out every 5th line by random chance
        sep = rc(["\n", "; ", ", "])
        address = sep.join(line for line in parts.values() if r() > 0.3)
        return address

    addresses = [gen_company_address() for i in tqdm.tqdm(range(size))]

    return addresses


# @functools.lru_cache(maxsize=5)
def get_address_collection():
    """

    this is how we are generating this collection:

    df = pd.read_csv(
        settings.trainingdir / "formatted_addresses_tagged.random.tsv",
        sep="\t",
        # nrows=100,
        names=["country", "lang", "address"],
        skiprows=lambda i: i > 0 and random.random() > 0.01
    )
    df.to_parquet(settings.trainingdir / "random_addresses.parquet")
    """

    df = pd.read_parquet(settings.DATADIR / "random_addresses.parquet")
    return df


def load_labeled_text_blocks(cached=True):
    filename = settings.TRAINING_DATA_DIR / "txtblocks.parquet"
    filename_html = settings.TRAINING_DATA_DIR / "html_text.parquet"
    label_file = settings.TRAINING_DATA_DIR / "labeled_txt_boxes.xlsx"
    if cached:
        try:
            df = pd.read_parquet(filename)
            return df
        except:
            logger.info("training data not available, recalculate")

    def regex_rand(choices):
        def rand_func(match_obj):
            return random.choice(choices)

        return rand_func

    df1 = pd.concat([
        get_pdf_text_boxes(extended=True),
        # TODO: label some addresess from our html blocks...
        pd.DataFrame(generate_random_textblocks(filename_html, iterations=10), columns=["txt"])
    ])
    # clean dataset from already labeled textboxes! we only want "unknown" text here
    df_labeled = pd.read_excel(label_file)
    addresses = df_labeled[df_labeled.label == "address"]
    df1.txt = df1.txt.str.strip()  # strip for (is this true?) better training results...
    df1 = df1.merge(
        addresses, on=['txt'], how="outer", suffixes=(None, "lbld"), indicator=True
    ).query('_merge=="left_only"')[df1.columns]  # finally filter out addresses
    df1['class'] = "unknown"

    # TODO: make sure to select only the languages that alsoappear in txtboxes in order
    #       to have a balanced dataset...
    df2 = get_address_collection()

    # langs = ("de", "us", "en")
    # df2 = df2[df2.lang.isin(langs)].copy()
    # make sure num of addresses equals num of textblocks to get a balanced
    # training set
    # in order to have a large number of addresses we triple the number
    multiplicator = 3
    if len(df2) > len(df1) * multiplicator:
        df2 = df2.sample(len(df1) * multiplicator)
    else:
        df1 = df1.sample(len(df2))

    logger.info("augment address database")
    # TODO: randomize FSEP and SEP fields for more training data
    # TODO: augment address with "labels" such as "addess:"  and "city" and
    #       "Postleitzahl", "country" etc...
    #       we can use a callable like this:
    #       def rand_func(match_obj):  and return a string based on match_obj
    # df2 = pd.concat([df2])
    fsep_repl = rand_chars({"\n": 4, ", ": 2, "; ": 1, " | ": 1})
    sep_repl = rand_chars({"\n": 4, ", ": 2, " ": 1})

    # df2 = df2.sample(20) # only for testing purposes
    def build_address(li):
        address = ""
        for add in li:
            parts = add.split("/")
            if parts[1] == "FSEP":
                address = address[:-1] + fsep_repl()
            elif parts[1] == "SEP":
                address = address[:-1] + sep_repl()
            else:
                address += parts[0] + " "
        return address[:-1]

    df2["txt"] = df2.address.str.split().progress_apply(build_address)

    logger.info("create fake addresses")
    training_addresses = fake_company_address_collection(
        fake_langs=['en_US', 'de_DE', 'en_GB'],
        size=len(df2) // 20
    )
    df2 = df2.append(pd.DataFrame(training_addresses, columns=["txt"]))
    df2['class'] = "address"

    # make sure short blocks are not recognized as addresses
    # 2 or fewer words
    shortaddr = df2.loc[df2['txt'].str.split().str.len() < 3]
    df2.loc[shortaddr.index, "class"] = "unknown"
    # less than 15 characters
    df2.loc[df2["txt"].str.len() <= 15, "class"] = "unknown"

    logger.info("putting together the final dataset")
    df = pd.concat([df1, df2])
    df = df.drop_duplicates(subset=["txt"])
    df = df.reset_index(drop=True)  # sample(frac=1) # mix the entire dataset
    df = df[["txt", "class"]]

    df.to_parquet(filename)
    return df


class PdfVectorDataSet(torch.utils.data.Dataset):
    def __init__(self, vectors, labels):
        self.vectors = vectors
        self.labels = labels

    def __getitem__(self, i):
        vector = self.vectors[i]
        return vector, self.labels[i]

    def __len__(self):
        return len(self.labels)


@functools.lru_cache()
def prepare_pdf_training():
    """
    loads textblock data, enriches it with addresses and
    other classes in order to train

    :return: pytorch dataloaders
    """

    logger.info("loading text blocks  dataset")
    df = load_labeled_pdf_files()

    classes = df['class'].cat.categories.tolist()
    classmap = dict(enumerate(classes))
    logger.info(f"extracted classes: {classmap}")

    model = pdfClassifier(classmap)
    model.string_vectorizer.init_embeddings()

    logger.info("pre-generating feature vectors for faster training")

    # def load_pdf_string
    model.cached = True
    X = model.generate_features(df.path.to_list())
    X = [x for x in X]  # convert into list of tensors

    Y = df['class'].cat.codes.tolist()  # pd.get_dummies(df.page_type)
    # Split into train+val and test
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, Y, test_size=0.2, random_state=69
    )

    train_dataset = PdfVectorDataSet(X_train, y_train)
    test_dataset = PdfVectorDataSet(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=64,
        num_workers=10
        # sampler=weighted_sampler
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=500,
        num_workers=5
        # sampler=weighted_sampler
    )
    return train_loader, test_loader, model


@functools.lru_cache()
def prepare_page_training():
    """
    Prepares training data, loaders and model parameters

    TODO: use the same function for generating the features here and
          in the classifier.

    """
    logger.info("loading dataset")
    df = load_labeled_webpages()

    classes = df.page_type.cat.categories.tolist()
    classmap = dict(enumerate(classes))
    logger.info(f"extracted classes: {classmap}")

    logger.info("get scaling for normalized metadata")
    x = df.progress_apply(lambda i: gen_meta_info_cached(i.url, i.raw_html), axis=1)
    x = np.log(x + 1.0)
    scaling = torch.stack(x.tolist()).max(0).values

    model = pageClassifier(
        scaling=scaling,
        classmap=classmap
    )
    model.string_vectorizer.init_embeddings()

    logger.info("pre-generating feature vectors for faster training")
    # "generate_features" only accepts lists but
    # in order to create beatches we need the individual feature
    # vectors not in a list
    X = df.progress_apply(
        lambda row: model.generate_features([row[['url', 'raw_html']]])[0], axis=1)

    X = X.tolist()
    # X = vecs.values
    Y = df.page_type.cat.codes.tolist()  # pd.get_dummies(df.page_type)
    # Split into train+val and test
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, Y, test_size=0.2, random_state=69
    )

    # return X_train, y_train

    class MyDataset(torch.utils.data.Dataset):
        def __init__(self, vecs, labels):
            self.vecs = vecs
            self.labels = labels

        def __getitem__(self, i):
            return self.vecs[i], self.labels[i]

        def __len__(self):
            return len(self.labels)

    train_dataset = MyDataset(X_train, y_train)
    test_dataset = MyDataset(X_test, y_test)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=64,
        num_workers=10
        # sampler=weighted_sampler
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=500,
        num_workers=5
        # sampler=weighted_sampler
    )
    return train_loader, test_loader, model


_asciichars = ''.join(sorted(set(chr(i) for i in range(32, 128)).union(string.printable)))


def random_string_augmenter(data: str, prob: float) -> str:
    return "".join(c if random.random() > prob else random.choice(_asciichars) for c in data)


class TextBlockDataset(torch.utils.data.Dataset):
    def __init__(self, rows, augment_prob: int = 0.0):
        """augmentation is mainly used if we are training..."""
        self.rows = rows.reset_index(drop=True)
        self.augment = augment_prob

    def __getitem__(self, i):
        # we use some augmentation for our dataset here to make the classifier more robust...
        # maybe include epsilon into our hyperparametr list?
        x, y = self.rows.loc[i][['txt', 'class_num']].to_list()
        # we don't need to return x as a "list", as our textblock model accepts a list of strings directly
        if self.augment:
            return random_string_augmenter(x, prob=self.augment), torch.tensor(y)
        else:
            return x, torch.tensor(y)

    def __len__(self):
        return len(self.rows)


@functools.lru_cache()
def prepare_textblock_training(num_workers: int = 4):
    """
    loads text block data and puts into pytorch dataloaders

    # TODO: this is the right place for hyperparameteroptimization....

    TODO: we also need to detect the pdf langauge in order
          to have a balances dataset with addresses from different countries
    """

    df = load_labeled_text_blocks()
    df['class_num'] = df['class'].astype("category").cat.codes.tolist()
    classes = df['class'].astype("category").cat.categories.tolist()
    classmap = dict(enumerate(classes))
    logger.info(f"extracted classes: {classmap}")
    model = txt_block_classifier(classmap)

    df_train, df_test = sklearn.model_selection.train_test_split(
        df, test_size=0.2, random_state=69
    )

    # give training dataset some augmentation...
    train_dataset = TextBlockDataset(df_train, augment_prob=1.0 / 50.0)
    test_dataset = TextBlockDataset(df_test)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=2 ** 9,
        num_workers=num_workers,
        # sampler=weighted_sampler
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=500,
        num_workers=num_workers
        # sampler=weighted_sampler
    )
    return train_loader, test_loader, model


@functools.lru_cache()
def prepare_url_training():
    df = load_labeled_webpages()
    # TODO make a classmap here instead of classes for added safety, that
    #      categorical encoding stay the same...
    classes = df.page_type.cat.categories.tolist()
    X = df.url
    y = df.page_type  # pd.get_dummies(df.page_type)
    # Split into train+val and test
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(
        X, y.cat.codes, test_size=0.2, random_state=69
    )

    class MyDataset(torch.utils.data.Dataset):
        def __init__(self, urls, labels):
            self.urls = urls
            self.labels = labels

        def __getitem__(self, i):
            return self.urls[i], self.labels[i]

        def __len__(self):
            return len(self.labels)

    train_dataset = MyDataset(X_train.tolist(), y_train.tolist())
    val_dataset = MyDataset(X_test.tolist(), y_test.tolist())
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=1,
        num_workers=10,
        # sampler=weighted_sampler
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=1,
        num_workers=10,
        # sampler=weighted_sampler
    )
    return train_loader, test_loader, classes


def train_url_classifier():
    train_loader, test_loader, classes = prepare_url_training()
    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)
    # trainer = pl.Trainer(gpus=8) (if you have GPUs)
    trainer = pytorch_lightning.Trainer(log_every_n_steps=100, max_epochs=300)

    trainer.fit(net, train_loader)
    trainer.save_checkpoint(settings.CLASSIFIER_STORE("url"))

    return trainer.test(net, test_dataloaders=test_loader, ckpt_path=settings.CLASSIFIER_STORE('url'))


def train_page_classifier(max_epochs=100, old_model=None):
    train_loader, test_loader, model = prepare_page_training()
    if old_model: model = old_model
    trainer = pytorch_lightning.Trainer(
        log_every_n_steps=100, max_epochs=max_epochs,
        checkpoint_callback=False
    )
    if True:
        # TODO: normally this should be discouraged (using test dataset for validation) ...
        #       but we don't have enough test data yet.
        val_dataloader = test_loader
        trainer.fit(model, train_loader, val_dataloader)
        trainer.save_checkpoint(settings.CLASSIFIER_STORE("page"))
        curtime = datetime.datetime.now()
        curtimestr = curtime.strftime("%Y%m%d%H%M")
        trainer.save_checkpoint(settings.CLASSIFIER_STORE("page").parent / f"pageClassifier{curtimestr}.ckpt")

    return trainer.test(model, test_dataloaders=test_loader, ckpt_path=settings.CLASSIFIER_STORE('page')), model


def train_text_block_classifier(old_model=None, num_workers=4, **kwargs):
    train_loader, test_loader, model = prepare_textblock_training(num_workers)
    if old_model:
        model = old_model
    checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(
        monitor='train_loss',  # or 'accuracy' or 'f1'
        mode='min', save_top_k=5,
        dirpath=settings.CLASSIFIER_STORE("text_block").parent,
        filename='text_block-{epoch:02d}-{train_loss:.2f}.ckpt'
    )
    trainer = pytorch_lightning.Trainer(
        accelerator="auto",  # "auto"
        gpus=kwargs.get("gpus", 1),
        # gpus=-1, auto_select_gpus=True,s
        log_every_n_steps=100,
        # limit_train_batches=100,
        max_epochs=kwargs.get("max_epochs", 100),
        # checkpoint_callback=False,
        enable_checkpointing=True,
        max_steps=-1,
        # auto_scale_batch_size=True,
        callbacks=[checkpoint_callback] + kwargs.get('callbacks', []),
        default_root_dir=settings.CLASSIFIER_STORE("text_block").parent
    )
    if True:
        # TODO: normally this should be discouraged to do this...
        #       (using test dataset for validation) ...
        #       but we don't have enough test data yet.
        val_dataloader = test_loader
        trainer.fit(model, train_loader, val_dataloader)
        trainer.save_checkpoint(settings.CLASSIFIER_STORE("text_block"))
        curtime = datetime.datetime.now()
        curtimestr = curtime.strftime("%Y%m%d%H%M")
        trainer.save_checkpoint(
            settings.CLASSIFIER_STORE("text_block").parent / f"text_blockclassifier{curtimestr}.ckpt")

    return trainer.test(model, test_dataloaders=test_loader, ckpt_path=settings.CLASSIFIER_STORE('text_block')), model


def train_pdf_classifier(max_epochs=100, old_model=None):
    train_loader, test_loader, model = prepare_pdf_training()
    if old_model: model = old_model
    trainer = pytorch_lightning.Trainer(
        log_every_n_steps=100, max_epochs=max_epochs,
        checkpoint_callback=False
    )
    if True:
        # TODO: normally this should be discouraged (using test dataset for validation) ...
        #       but we don't have enough test data yet.
        val_dataloader = test_loader
        trainer.fit(model, train_loader, val_dataloader)
        trainer.save_checkpoint(settings.CLASSIFIER_STORE("pdf"))
        curtime = datetime.datetime.now()
        curtimestr = curtime.strftime("%Y%m%d%H%M")
        trainer.save_checkpoint(settings.CLASSIFIER_STORE("pdf").parent / f"pdfClassifier{curtimestr}.ckpt")

    return trainer.test(model, test_dataloaders=test_loader, ckpt_path=settings.CLASSIFIER_STORE('pdf')), model


def train_fqdn_componentsearch_classifier(samplenum=1000):
    """
    this classifier can estimate the potential to
    find component within this base-url.

    """
    NotImplementedError("has to be rewritten for pytorch")
    # TODO: we could directly take the vectors from the database using pandas...
    # OR we could export the data into some other format from the Rest Interface
    # OR OR OR ... ;)
    logger.info("loading page vectors")
    df = load_page_vectors()

    logger.info("classifying pages")
    # classify all webpages to check for components
    page_classifier = load_pipe()
    embeddings = np.stack(df.embeddings)
    prediction = predict_proba(page_classifier, embeddings)
    prediction.index = df.index
    df = df.join(prediction)

    logger.info("extract fully qualified domain names")
    # and aggregate their respective number of component pages
    NotImplementedError("replace tldextract with urllib")
    df['fqdn'] = df.url.apply(lambda x: tldextract.extract(x).fqdn)
    componentnum = df.groupby('fqdn').apply(
        lambda x: (x['class'] == 'component').sum()
    ).rename('componentnum')

    df = df.merge(componentnum, how='left', left_on='fqdn',
                  right_index=True)

    df['fqdn_has_components'] = df['componentnum'].apply(
        lambda x: "has_components" if x > 0 else "no_components")
    trainingset = df.query("`class`!='component'")
    if samplenum < len(trainingset) and samplenum > 0:
        trainingset = trainingset.sample(samplenum)

    X = np.stack(trainingset['embeddings'])
    Y = trainingset['fqdn_has_components']

    (X_train, X_test,
     y_train, y_test) = sk.model_selection.train_test_split(X, Y,
                                                            test_size=0.3,
                                                            random_state=4)

    class_weights = sk.utils.class_weight.compute_class_weight(
        'balanced',
        y_train.unique(),
        y_train)

    class_weights = dict(zip(y_train.unique(), class_weights))
    classifier = sk.linear_model.LogisticRegression(
        max_iter=200, class_weight=class_weights,
        solver='saga',
        verbose=1,
        n_jobs=-1)
    logger.info("train classifier")
    classifier.fit(X_train, y_train)

    test_classifier(classifier, X_test, y_test)

    logger.info("saving pipeline")
    fn = settings.FQDN_COMPONENTSEARCH_CLASSIFIER_FILE
    with open(fn, 'wb') as handle:
        pickle.dump(classifier, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return classifier
