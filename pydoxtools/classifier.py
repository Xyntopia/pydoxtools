#!/usr/bin/env python3
"""
Created on Mon Apr  6 22:01:10 2020
TODO write file description
"""
from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import abc
import functools
import logging
import pathlib
import typing
import urllib
from typing import Tuple, List

import numpy as np
import pandas as pd
import pytorch_lightning
import sklearn
import torch
import torchmetrics
import tqdm
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel

from pydoxtools import html_utils
from pydoxtools.settings import settings

logger = logging.getLogger(__name__)
tqdm.tqdm.pandas()


def init_embeddings(model_name: str = None, dimension=None):
    """
    This function can be used to pre-initialize the embeddings using
    for example BERT embeddings.

    We freeze the embeddings used here so that we can generalize the vectorization
    for many documents and tasks. this way we can generate the vectorizations
    once and then store them in a database and reuse them for all kinds of different tasks.

    """
    # TODO: add umap embeddings conversion "on-the-fly"
    if model_name:
        model = AutoModel.from_pretrained(model_name)
        weights = model.embeddings.word_embeddings.weight
    else:
        # load pre-initialized embeddings
        weights = torch.tensor(pd.read_parquet(settings.TRAINING_DATA_DIR / "red_bert_emb_64.parquet").to_numpy())

    embedding = torch.nn.Embedding.from_pretrained(
        weights,
        freeze=False, padding_idx=None, max_norm=None,
        norm_type=2.0, scale_grad_by_freq=False, sparse=False
    )
    return embedding


class string_vectorizer(torch.nn.Module):
    """
    This module takes a webpage and finds an embedding using
    a pre-generated look-up table.56
    """

    def __init__(self):
        super().__init__()
        self.model_name = settings.PDXT_STANDARD_TOKENIZER  # name of model used for initialization
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        embedding_num, embedding_dim = 119547, 768  # size of multilingual BERT vocabulary
        self.embedding = torch.nn.Embedding(embedding_num, embedding_dim)

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


# TODO: cache this with diskcache
def cached_str_vectorize(model, raw_string):
    return model(raw_string)


def make_tensorboard_compatible_dict(d, prefix=""):
    new_d = {}
    for k, v in d.items():
        if isinstance(v, dict):
            sub_d = make_tensorboard_compatible_dict(v, prefix=k + ".")
            new_d.update(sub_d)
        else:
            new_d[prefix + k] = torch.tensor(v, dtype=torch.float32)
    return new_d


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
        super().__init__()
        self.histograms = True
        self._hp_metric = None

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
            'f1': torchmetrics.F1Score(num_classes, threshold=threshold),
            'recall': torchmetrics.Recall(num_classes, threshold=threshold),
            'precision': torchmetrics.Precision(num_classes, threshold=threshold)
        })

    def training_step(self, batch, batch_idx):
        # TODO: add option to remember "failed" predictions, cache them
        #      and repeat their training a fixed number of times!
        # TODO: use f1-accuracy as loss!
        # training_step defined the train loop.
        # It is independent of forward
        x, y = batch
        y_train_pred = self(x)

        # this function handles the multiple categorical outputs from
        # y_train_pred and numerical categories from y automatically...
        loss = F.cross_entropy(
            y_train_pred, y,
            # weight=self.class_weight.to(device=self.device)
        )
        # Logging to TensorBoard by default
        # self.log('train_loss', loss)
        self.log(
            'train_loss', loss, on_step=True,
            on_epoch=True, prog_bar=True, logger=True,
            batch_size=len(batch),  # sync_dist=True
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
        return self.validation_epoch_end(test_step_outputs)

    def validation_step(self, batch, batch_idx):
        return self.test_step(batch, batch_idx)

    def validation_epoch_end(self, val_step_outputs):
        pred, target = list(zip(*val_step_outputs))
        pred, target = torch.cat(pred), torch.cat(target)
        for metric_name in self.metrics:
            self.metrics[metric_name](pred, target)
            # self.log(metric_name, self.metrics[metric_name], sync_dist=True)

        classes = list(self.classmap_.values())
        est = pd.DataFrame(pred.cpu().numpy(), columns=classes)
        est['target'] = target.cpu().numpy()
        est['max'] = est.loc[:, classes].max(axis=1)  # get the calculated certainty
        est['maxidx'] = est.loc[:, classes].idxmax(axis=1)  # get the label
        est['target_label'] = est['target'].map(self.classmap_)

        # Model Accuracy
        predicted, test = est[['maxidx', 'target_label']].values.T
        # accuracy = sklearn.metrics.accuracy_score(test, predicted)

        classification_report = sklearn.metrics.classification_report(
            test, predicted,
            output_dict=True,
            zero_division=0
        )

        metrics = make_tensorboard_compatible_dict(classification_report)
        self.log_dict(metrics)  # , sync_dist=True)
        if self._hp_metric:
            self.log("hp_metric", metrics[self._hp_metric], prog_bar=True)
        return metrics

    def configure_optimizers(self):
        weight_decay = 0.0
        learning_rate = self.hparams.learning_rate
        optimizer = torch.optim.Adam(
            self.parameters(), lr=learning_rate,
            weight_decay=weight_decay
        )
        return optimizer


def get_topk_ngrams(txt: str, k: int = 5, ngram_size: int = 2):
    """
    generate top k frequency ngrams for a string

    This can help in identifying the structure of a text. For example the top frequency ngrams
    of a table will be very high count while in a "normal" text they will be very low. Random
    letters will be the lowest.
    """
    txt = txt.lower()
    ngrams = list(zip(*[txt[i:] for i in range(ngram_size)]))
    unique, counts = np.unique(ngrams, return_counts=True, axis=0)  # count ngrams
    # get top k elements. High numbers should indicate high repetitiveness
    ind = np.argpartition(counts, -min(counts.shape[0], k))[-k:]
    topk = counts[ind]
    topk = np.pad(topk, (0, k - len(topk)), 'constant', constant_values=0)
    topk = np.sort(topk)
    return topk


class txt_block_classifier(
    lightning_training_procedures,
    classifier_functions,
    pytorch_lightning.LightningModule
):
    """
    This model takes a string and classifies it.
    """

    def __init__(
            self, classmap,
            embeddings_dim=2,  # embeddings vector size (standard BERT has a vector size of 768 )
            token_seq_length1=5,  # what length of a work do we assume in terms of tokens?
            seq_features1=40,  # how many filters should we run for the analysis = num of generated features?
            dropout1=0.3,  # first layer dropout
            cv_layers=2,  # number of cv layers
            token_seq_length2=40,  # how many tokens in a row do we want to analyze?
            seq_features2=100,  # how many filters should we run for the analysis?
            dropout2=0.3,  # second layer dropout
            fft_pool_size=20,  # precision of fft for spectral pooling at the end
            learning_rate=0.01,
            fft_pooling=True,
            ngram_size=2,  # size of ngrams to calculate for features
            topk_ngrams=5,  # number of top frequency ngrams to include in metadata
            meanmax_pooling=True,
            hp_metric=None,
            embeddings_mode=None
    ):
        super().__init__()

        # we are saving all hyperparameters from above in the model checkpoint this way...
        self.save_hyperparameters()
        self.classmap_ = self.hparams.classmap
        self.classmapinv_ = {v: k for k, v in classmap.items()}
        self.classes_ = list(classmap.values())
        num_classes = len(classmap)
        self._hp_metric = self.hparams.hp_metric

        # declare our embeddings we use the size of multilingial BET vocabulary
        # as we are using the same tokenizer
        # size of multilingual BERT vocabulary this should not be changed, as it is defined by the tokenizer!
        embedding_num = 119547

        # TODO: the 2D paramters of the AvgPool would be a really nice parameter
        #       for hyperparameter optimization
        # TODO: checkout if it might be a better idea to use transformers for this task?
        # TODO: finetune a hugginface transformer for this task?
        if embeddings_dim:
            if self.hparams.embeddings_mode == "bert":
                self.embedding = init_embeddings(model_name="bert-base-multilingual-cased")
                # reduce vector size
                # self.l1 = torch.nn.Linear(in_features=embedding_dim, out_features=embeddings_dim)
                self.cv0 = torch.nn.Conv2d(
                    in_channels=1, out_channels=self.hparams.embeddings_dim,
                    kernel_size=(1, self.embedding.embedding_dim),  # reduce size of word vectors
                    stride=1
                )
            elif self.hparams.embeddings_mode == "pre_initialized":
                self.embedding = init_embeddings(dimension=self.hparams.embeddings_dim)
            elif self.hparams.embeddings_mode == "fresh":
                self.embedding = torch.nn.Embedding(embedding_num, self.hparams.embeddings_dim)

            self.dropout1 = torch.nn.Dropout(p=self.hparams.dropout1)
            self.cv1 = torch.nn.Conv2d(
                in_channels=1,  # only one layer of embeddings
                out_channels=self.hparams.seq_features1,  # num of encoded features/word
                kernel_size=(
                    self.hparams.token_seq_length1,
                    self.hparams.embeddings_dim
                    # we want to put all features of the embedded word vector through the filter
                ),
                # the second dimension of stride has no effect, as our filter has the same size
                # as the vector anyways and we can leave it at 1
                # we are dividing the stride by two to make sure our scan filters overlap by 50%
                stride=(self.hparams.token_seq_length1 // 2, 1)
            )
            self.dropout2 = torch.nn.Dropout(p=self.hparams.dropout2)
            # TODO: we might no need to take ALL features into acount
            # every single time so maybe make the kernel_size smaller than seq_features1
            if self.hparams.cv_layers > 1:
                self.cv2 = torch.nn.Conv2d(
                    in_channels=1,  # only one layer of embeddings
                    out_channels=self.hparams.seq_features2,  # num of encoded features/word
                    kernel_size=(
                        # we have to switch around features and seq length, as cv1 puts
                        # the kernel features before the new sequence length. This is, because
                        # we are treating embeddings vector dimension as number of "features"
                        # as input for cv1. but the number of features for the vectors are in the 3d dimension whereas
                        # in cv we need have them as the second dimension due to their variable length.
                        # so cv1 switches the dimensions and onwards of cv2 we need to switch features and seq_length
                        # in comparison to cv1.
                        self.hparams.seq_features1,
                        self.hparams.token_seq_length2
                    ),
                    # this time we have to switch around the stride as we now have the results
                    # from the previous layer in the other dimension
                    # ultimately it doesn't matter which way we organize the features s we are
                    # moving 2D-filters over them anyways...
                    # we are dividing the stride by two to make sure our scan filters overlap by 50%
                    # TODO: use stride length as hyperparameter...
                    stride=(1, self.hparams.token_seq_length2 // 2)
                )

        # afterwards we do a max pooling and feed the input into a linear layer
        # we add addtional features here...
        # right now: 1: length of string + number of lines + number of words (defined by spaces)
        #                   + number of digits + topk_ngrams
        # TODO: add number of seperators, number of letters, number of digit-words
        meta_features = 1 + 1 + 1 + 1 + self.hparams.topk_ngrams
        final_cv_feature_num = self.hparams.seq_features2 if self.hparams.cv_layers > 1 else self.hparams.seq_features1
        self.fft_pool_size = (final_cv_feature_num, self.hparams.fft_pool_size)
        fft_out = (self.hparams.fft_pool_size // 2 + 1) if self.hparams.fft_pooling else 0
        pooling_out = 2 if self.hparams.meanmax_pooling else 0
        if embeddings_dim:
            linear1_features = final_cv_feature_num * (fft_out + pooling_out) + meta_features
        else:
            linear1_features = meta_features

        linear_dims = [100, 50, num_classes]
        self.linear = torch.nn.ModuleList()
        in_dem = linear1_features
        for out_dim in linear_dims:
            self.linear.append(torch.nn.Linear(
                in_features=in_dem,
                out_features=out_dim
            ))
            in_dem = out_dim

        # number of features can roughly be calculated ike this:
        # param_num = embedding_num * embeddings_dim
        # + ((token_seq_length1 * embeddings_dim + 1) * seq_features1)
        # + ((token_seq_length2 * seq_features1 + 1) * seq_features2)
        # + (seq_features2 * (((fft_pool_size // 2) + 1) + 2) + 3 + 1) * num_classes

        # and add metrics
        self.add_standard_metrics(num_classes, hist_params=[
            "linear.weight", "linear.bias",
            "cv2.weight", "cv2.bias",
            "cv1.weight", "cv1.bias"
        ])

    @functools.cached_property
    def tokenizer(self):
        # TODO: get rid of model dependency... only use the vocabulary for the tokenizer...
        self.model_name = settings.PDXT_STANDARD_TOKENIZER  # name of model used for initialization
        return AutoTokenizer.from_pretrained(self.model_name)

    def forward(self, str_list: typing.Iterable):
        # TODO: we might be able to us "diskcache" here in order to cache some layers!!!
        #       for example the tokenized ids
        # count number of letters, words and lines here...
        str_lens = torch.tensor([len(s) for s in str_list]).unsqueeze(1)
        counts = torch.tensor([(
            s.count("\n"), s.count(" "),
            sum(c.isdigit() for c in s),
            # create n-grams and count their frequency. Add frequency of top-5 n-grams to
            # metadata. The hope is that we can recognize repetitiveness more easily this way.
            *get_topk_ngrams(s, k=self.hparams.topk_ngrams, ngram_size=self.hparams.ngram_size)
        ) for s in str_list], dtype=torch.float32).unsqueeze(1).to(device=self.device).squeeze()
        # normalize meta data on string length
        counts_norm = counts / (str_lens + 0.0001)  # adding 0.0001 to prevent div errors
        # len(s) / 2000,  # normalize size with 2000 (approx 1 pages = 1.0)
        meta = torch.cat([counts_norm, str_lens / 2000], 1)

        if self.hparams.embeddings_dim:
            # TODO: add optional str.strip here... (deactivated during training to improve speed)
            ids = self.tokenizer(
                list(str_list),  # convert trlist into an actual list (e.g. if it is a pandas dataframe)
                padding='max_length', truncation=True,
                max_length=500,  # approx. half an A4 page of text
                return_tensors='pt',
                return_length=True  # use this as another feature
            ).input_ids.to(device=self.device)
            # x = torch.Tensor(ids.shape[0], ids.shape[1])
            # ids = torch.tensor(ids)
            x = self.embedding(ids)
            if self.hparams.embeddings_mode == "bert":
                x = self.cv0(x.unsqueeze(1)).squeeze(3).transpose(1, 2)  # reduce size of embeddings
            x = self.dropout1(x)
            # TODO: implement more fft directly after initialization
            # x_fft = torch.fft.rfft2(x, s=self.fft_pool_size).real
            # x = self.l1(x) # if we use "frozen" embeddings we want to transform them into a more usable format...
            # "unsqueeze" to add the channel dimensions and then "squeeze" out the channel at the end...
            x = self.cv1(x.unsqueeze(1)).squeeze(3)
            x = self.dropout2(x)
            if self.hparams.cv_layers > 1:
                x = self.cv2(x.unsqueeze(1)).squeeze(2)
        # TODO: look at the text in "different resolution" by applying multiple conv2d using
        #       different sequence lengths/strides
        # TODO: does it make sense to introduce another feature layer to extract even more complexity?
        #       do some hyperparameter reaserch on this...
        # TODO: specify pooling method as a hypeparameter as well...
        # max:  takes the max of every feature vector (meaning the max value for
        # every individual feature for the entire text). hats left over are 1 value for each feature
        poolings = [meta]
        if self.hparams.embeddings_dim:
            if self.hparams.meanmax_pooling:
                poolings.append(torch.max(x, dim=2).values)
                poolings.append(torch.mean(x, dim=2))
            if self.hparams.fft_pooling:
                x_fft = torch.fft.rfft2(x, s=self.fft_pool_size).real
                poolings.append(x_fft.flatten(1))

        x = torch.cat(poolings, 1)
        # combine features
        # finally, interprete the features
        x = x.flatten(1)
        for linear in self.linear:
            x = linear(x)
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
        super().__init__()
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
        super().__init__()
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

    TODO: turn this into a document classifier...
    TODO: move this into the document class itself...
    """
    __version__ = "txt20200807"

    def __init__(self, classmap):
        super().__init__()
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
        # TODO: replace this with "document" or som generic text
        pdftxtfunc = pdf_utils.get_pdf_text_safe_cached if self.cached else pdf_utils.get_pdf_text_safe
        pdfmeta = pdf_utils.get_meta_infos_safe_cached if self.cached else pdf_utils.get_meta_infos_safe
        for pdf in pdf_list:
            try:
                filename = pdf.name
                filename_vec = self.string_vectorizer(filename)
                # TODO: get rid of pdftxtfunc and get text directly from document
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


@functools.lru_cache
def load_classifier(name):
    logger.info(f"loading classifier: {name}")
    if name == "text_block":
        net = txt_block_classifier.load_from_checkpoint(settings.MODEL_STORE(name))
    elif name == "url":
        net = urlClassifier.load_from_checkpoint(settings.MODEL_STORE(name))
    elif name == "page":
        net = pageClassifier.load_from_checkpoint(settings.MODEL_STORE(name))
    elif name == "pdf":
        net = pdfClassifier.load_from_checkpoint(settings.MODEL_STORE(name))

    net.freeze()
    return net


@functools.lru_cache
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



# TODO: move all the following functions into a separate
#       "data preparation" file and only load the generated data here...


# @functools.lru_cache(maxsize=5)


class CLassificationExtractor():
    """Operator which ca classify """
    # TODO: make this class a lot more configurable ;)
    pass
