from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import datetime
import functools
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning
import sklearn
import torch

from pydoxtools import file_utils
from pydoxtools.classifier import pdfClassifier, gen_meta_info_cached, \
    pageClassifier, txt_block_classifier
from pydoxtools.random_data_generators import TextBlockGenerator
from pydoxtools.settings import settings

logger = logging.getLogger(__name__)


@functools.lru_cache
def load_labeled_webpages():
    trainingfiles = file_utils.get_nested_paths(settings.TRAINING_DATA_DIR, "trainingitem*.parquet")

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


@functools.lru_cache
def load_labeled_pdf_files(label_subset=None) -> pd.DataFrame:
    """
    This function loads the pdfs from the training data and
    labels them according to the folder they were found in...

    TODO: use the label_subset
    """

    # TODO: remove settings.trainingdir and movethis to "analysis" or somewhere else...
    df = file_utils.get_nested_paths(settings.TRAINING_DATA_DIR, '*.pdf')
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


# TODO: make sure rand_chars can abide to seeds everywhere where its used...!!


class PdfVectorDataSet(torch.utils.data.Dataset):
    def __init__(self, vectors, labels):
        self.vectors = vectors
        self.labels = labels

    def __getitem__(self, i):
        vector = self.vectors[i]
        return vector, self.labels[i]

    def __len__(self):
        return len(self.labels)


@functools.lru_cache
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


@functools.lru_cache
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


class PandasDataset(torch.utils.data.Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self._X = X
        self._y = y

    def __getitem__(self, i):
        return self._X[i], self._y[i]

    def __len__(self):
        return len(self._X)


def load_labeled_text_block_data(classmap: dict = None):
    """load evaluation data for textblock classification"""
    # load manually labeled dataset for performance evaluation
    label_file = settings.TRAINING_DATA_DIR / "labeled_txt_boxes.xlsx"
    df = pd.read_excel(label_file).fillna(" ")
    y = df["label"].replace(dict(
        contact="unknown",
        directions="unknown",
        company="unknown",
        country="unknown",
        url="unknown",
        list="unknown",
        multiaddress="unknown",
        phone_number="unknown"
    ))
    if classmap:
        y = y.replace(classmap)
        assert y.dtype == np.dtype('int64'), f"has to be numeric!! y.unique: {y.unique()}"
    return df, y


def prepare_textblock_training(
        num_workers: int = 4,
        steps_per_epoch: int = 500,
        batch_size=2 ** 10,
        data_config=None,
        model_config=None,
        model_name: str = None,
        **kwargs
):
    """
    prepare training infrastructure for txtblock classifier
    """
    # label_file = settings.TRAINING_DATA_DIR / "labeled_txt_boxes.xlsx"
    # df_labeled = pd.read_excel(label_file)

    virtual_size = steps_per_epoch * batch_size
    user_generator_config = dict(virtual_size=virtual_size)
    user_generator_config.update(data_config or {})
    train_dataset = TextBlockGenerator(**user_generator_config)

    x, y = load_labeled_text_block_data(classmap=train_dataset.classmap_inv)
    validation_dataset = PandasDataset(X=x['txt'], y=y)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True
        # sampler=weighted_sampler
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset=validation_dataset,
        batch_size=len(validation_dataset),
        num_workers=num_workers,
        persistent_workers=True
        # sampler=weighted_sampler
    )

    classmap = train_dataset.classmap
    logger.info(f"extracted classes: {classmap}")
    model = txt_block_classifier(classmap, **(model_config or {}))

    from pytorch_lightning.loggers import TensorBoardLogger

    trainer = pytorch_lightning.Trainer(
        fast_dev_run=kwargs.get('fast_dev_run', False),
        accelerator=kwargs.get('accelerator', 'cpu'),
        devices=kwargs.get('devices', 1),
        strategy=kwargs.get('strategy', 'ddp'),
        # gpus=-1, auto_select_gpus=True,s
        log_every_n_steps=kwargs.get("log_every_n_steps", 100),
        logger=pytorch_lightning.loggers.TensorBoardLogger(
            save_dir=settings.MODEL_STORE("text_block").parent,
            version=model_name,
            name="lightning_logs"
        ),
        # limit_train_batches=100,
        max_epochs=kwargs.get("max_epochs", -1),
        # checkpoint_callback=False,
        enable_checkpointing=kwargs.get("enable_checkpointing", False),
        max_steps=-1,
        # auto_scale_batch_size=True,
        callbacks=kwargs.get('callbacks', []),
        default_root_dir=settings.MODEL_STORE("text_block").parent
    )

    return train_loader, validation_loader, model, trainer


@functools.lru_cache
def prepare_url_training():
    # TODO:  use our textblock classifier for this!
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
    trainer.save_checkpoint(settings.MODEL_STORE("url"))

    return trainer.test(net, test_dataloaders=test_loader, ckpt_path=settings.MODEL_STORE('url'))


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
        trainer.save_checkpoint(settings.MODEL_STORE("page"))
        curtime = datetime.datetime.now()
        curtimestr = curtime.strftime("%Y%m%d%H%M")
        trainer.save_checkpoint(settings.MODEL_STORE("page").parent / f"pageClassifier{curtimestr}.ckpt")

    return trainer.test(model, test_dataloaders=test_loader, ckpt_path=settings.MODEL_STORE('page')), model


def train_text_block_classifier(
        train_loader,
        validation_loader,
        model,
        trainer,
        log_hparams: dict = None,
        old_model: txt_block_classifier = None,
        save_model=False,
        checkpoint_path=None
):
    # only importing this here as we don#t want to use this in "inference" mode

    if old_model:
        model = old_model

    # TODO: ho can we set hp_metric to 0 at the start?
    # if we use an already initialized "start_model" we can not log hyperparameters!!
    trainer.logger.log_hyperparams(log_hparams)  # , metrics=dict(hp_metric=0))
    trainer.fit(
        model, train_loader, validation_loader,
        ckpt_path=checkpoint_path
    )
    # trainer.logger.log_hyperparams(log_hparams, metrics=dict(
    #    model.metrics
    # ))
    if save_model:
        trainer.save_checkpoint(settings.MODEL_STORE("text_block"))
        trainer.save_checkpoint(settings.MODEL_STORE("text_block_wo"), weights_only=True)
        curtime = datetime.datetime.now()
        curtimestr = curtime.strftime("%Y%m%d%H%M")
        trainer.save_checkpoint(
            settings.MODEL_STORE(
                "text_block").parent / f"text_blockclassifier_{trainer.logger.version}_{curtimestr}.ckpt")

    return trainer, model


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
        trainer.save_checkpoint(settings.MODEL_STORE("pdf"))
        curtime = datetime.datetime.now()
        curtimestr = curtime.strftime("%Y%m%d%H%M")
        trainer.save_checkpoint(settings.MODEL_STORE("pdf").parent / f"pdfClassifier{curtimestr}.ckpt")

    return trainer.test(model, test_dataloaders=test_loader, ckpt_path=settings.MODEL_STORE('pdf')), model


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
