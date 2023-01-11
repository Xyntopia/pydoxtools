# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import typing
import traceback

# %%
# study_params
# select GPU device: CUDA_VISIBLE_DEVICES=1,2
# run with:
# CUDA_VISIBLE_DEVICES=2 TOKENIZERS_PARALLELISM=true python /project/analysis/txtblock_classification/train_txtblock_classifier_hyper.py 1

study_name = "hyparams_50max2"
optimize = True
# if we use a "start_model" we will not have hyperparameters!!
# also: if some model in tensorboard are using hyperparameters, while
#       others don't, we will not have hyperparameters displayed!!
start_model = ""  # text_blockclassifier_x0.ckpt"
max_mb = 50

epoch_config = dict(
    steps_per_epoch=100,
    log_every_n_steps=20,
    max_epochs=20
)
epoch_config1 = dict(
    steps_per_epoch=2,
    log_every_n_steps=1,
    max_epochs=2
)


# %%
import datetime
import logging
import sys
import platform

import optuna
import pytorch_lightning
import torch
from IPython.display import display, HTML

# %% tags=[]
# %load_ext autoreload
# %autoreload 2
# from pydoxtools import nlp_utils
from pydoxtools import pdf_utils, nlp_utils, cluster_utils, training, classifier
from pydoxtools import webdav_utils as wu
from pydoxtools.settings import settings


# %% [markdown] tags=[]
# # Train the Textblock classifier
#
# import datetime
# %% tags=[]
def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "<br>")))


logger = logging.getLogger(__name__)

box_cols = cluster_utils.box_cols

pdf_utils._set_log_levels()
memory = settings.get_memory_cache()

nlp_utils.device, torch.cuda.is_available(), torch.__version__, torch.backends.cudnn.version()

# %% [markdown] tags=[]
# test the model once

# %%
if False:
    _, _, m, _ = training.prepare_textblock_training()
    res = m.predict(["""ex king ltd
    Springfield Gardens
    Queens
    N. Y 11413
    www.something.com
    """
                     ])
    print(res)

# %% [markdown]
# TODO: its probabybl a ood idea to use some hyperparemeter optimization in order to find out what is the best method here...
#
# we would probably need some manually labeled addresses from our dataset for this...

# %% [markdown]
# start training

# %% tags=[]
# %env TOKENIZERS_PARALLELISM=true
# url of nextcloud instance to point to
hostname = 'https://sync.rosemesh.net'
# the token is the last part of a sharing link:
# https://sync.rosemesh.net/index.php/s/KwkyKj8LgFZy8mo   the  "KwkyKj8LgFZy8mo"  webdav
# takes this as a token with an empty password in order to share the folder
token = "KwkyKj8LgFZy8mo"
syncpath = str(settings.MODEL_DIR)
upload = False

# %%
# test webdav connection
settings.MODEL_DIR.mkdir(parents=True, exist_ok=True)
# and create a timestamp file to make sure we know it works!


# %%
ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
sysinfo = dict(
    platform=platform.platform(),
    cpu=platform.processor()
)
with open(settings.MODEL_DIR / f"ts_{ts}.txt", "w") as f:
    f.write(str(sysinfo))

# %%
# we don't need this right now, as our modelnames get synchronized through the optuna msql storage...
# wu.rclone_single_sync_models(method="bisync", hostname=hostname, token=token, syncpath=syncpath)
# # copy our start model to the new training process
if start_model:
    wu.rclone_single_sync_models(
        method="copyto", hostname=hostname, token=token,
        syncpath=syncpath, file_name=start_model, reversed=True)


# %% tags=[]
# %env TOKENIZERS_PARALLELISM=true
class WebdavSyncCallback(pytorch_lightning.Callback):
    def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        #print("""lightning: sync models with rclone!""")
        wu.rclone_single_sync_models(method="copy", hostname=hostname, token=token, syncpath=syncpath)


additional_callbacks: list[typing.Any] = [
    WebdavSyncCallback(),
    # pytorch_lightning.callbacks.RichProgressBar()
]

import warnings

warnings.filterwarnings("ignore", ".*Your `IterableDataset` has `__len__` defined.*")


# we introduce a report call back in order to stop optimization runs early
class ReportCallback(pytorch_lightning.Callback):
    def __init__(self, trial: optuna.trial.BaseTrial):
        self._trial = trial

    def on_train_epoch_end(self, trainer: pytorch_lightning.Trainer, pl_module: "pl.LightningModule") -> None:
        score = trainer.callback_metrics['address.f1-score']
        # TODO: only works if we are optimizing for a single objective!
        try:
            self._trial.report(score, trainer.current_epoch)
        except NotImplementedError:
            # trying to catch multi-objective optimization
            # where the report function doesn't work!
            logger.info("not reporting, as we are probably going for multi-objective optimization...")


def train_model(trial: optuna.trial.BaseTrial):
    additional_callbacks.append(ReportCallback(trial))
    data_config = dict(
        generators={
            "address": (
                (100, training.BusinessAddressGenerator(
                    rand_str_perc=0.5,  # trial.suggest_float("rand_str_perc", 0.1, 0.4),
                    osm_perc=0.5,
                    fieldname_prob=0.05)),
            ),
            "unknown": ((50, training.RandomTextBlockGenerator()), (50, training.RandomListGenerator()))
        },
        cache_size=20000,
        renew_num=2000,
        random_separation_prob=0.2,
        random_line_prob=0.1,
        random_char_prob=0.05,  # trial.suggest_float("random_char_prob", 0.0, 1.0),
        random_word_prob=0.1,  # trial.suggest_float("random_word_prob", 0.0, 1.0),
        random_upper_prob=0.3,  # trial.suggest_float("random_upper_prob", 0.0, 1.0),
        mixed_blocks_generation_prob=0.05,  # trial.suggest_float("mixed_blocks_generation_prob", 0.05, 0.1),
        mixed_blocks_label="unknown",
    )

    model_config = dict(
        learning_rate=0.0005,
        embeddings_dim=trial.suggest_int("embeddings_dim", 1, 32),
        # embeddings vector size (standard BERT has a vector size of 768 )
        token_seq_length1=trial.suggest_int("token_seq_length1", 3, 16),
        # what length of a word do we assume in terms of tokens?
        seq_features1=trial.suggest_int("seq_features1", 10, 500),
        # how many filters should we run for the analysis = num of generated features?
        dropout1=0.5,  # first layer dropout
        cv_layers=trial.suggest_int("cv_layers", 1, 2),  # number of cv layers
        token_seq_length2=trial.suggest_int("token_seq_length2", 3, 100),
        # how many tokens in a row do we want to analyze?
        seq_features2=trial.suggest_int("seq_features2", 10, 500),  # how many filters should we run for the analysis?
        dropout2=0.5,  # second layer dropout
        meanmax_pooling=trial.suggest_int("meanmax_pooling", 0, 1),  # whether to use meanmax_pooling at the end
        fft_pooling=trial.suggest_int("fft_pooling", 0, 1),  # whether to use fft_pooling at the end
        fft_pool_size=trial.suggest_int("fft_pool_size", 5, 50),  # size of the fft_pooling method
        hp_metric="address.f1-score"  # the metric to optimize for and should be logged...
    )

    if start_model:
        m = classifier.txt_block_classifier.load_from_checkpoint(
            settings.MODEL_DIR / start_model)
    else:
        m = None

    train_loader, validation_loader, model, trainer = training.prepare_textblock_training(
        model_name=trial.number,
        num_workers=4,
        data_config=data_config, model_config=model_config,
        # strategy="ddp_find_unused_parameters_false",
        # strategy="ddp",
        enable_checkpointing=False,
        strategy=None,  # in case of running jupyter notebook
        callbacks=additional_callbacks,
        batch_size=2 ** 12,
        accelerator="auto", devices=1,  # use simple integer to specify number of devices...
        **epoch_config,
    )

    model_mb = pytorch_lightning.utilities.memory.get_model_size_mb(model)  # minimize
    # Store the constraints as user attributes so that they can be restored after optimization.
    constraint = [model_mb - max_mb]  # constraint <=0  are considered as feasible!
    trial.set_user_attr("constraint", constraint)

    print(f"entering: {trial.params}")

    if model_mb > (max_mb + 1):  # basically our constraint!
        print(f"model has {model_mb}>{max_mb}+1 => fast-return!!")
        metric = 0
    else:
        try:
            trainer, model = training.train_text_block_classifier(
                train_loader=train_loader, validation_loader=validation_loader,
                model=model, trainer=trainer,
                log_hparams=trial.params,
                old_model=m
            )
            metric = trainer.callback_metrics['address.f1-score']  # maximize
        except Exception as ex:
            logger.exception("training failed!!")
            tb = "".join(traceback.TracebackException.from_exception(ex).format())
            trial.set_user_attr("exception", tb)
            metric = 0
        # trainer.logger.log_hyperparams()
        # calculate score

    objectives = metric, model_mb
    return objectives  # maximize, minimize


# %%
local_storage = f"sqlite:///{str(settings.MODEL_DIR)}/study.sqlite"
remote_storage = "TODO: get from env variable (f"mysql+pymysql:....")"
remote_storage


# %% [markdown]
# create a mysql server in docker like this:
#
#
# example docker-compose.yml file:
#
# ```
# version: '3' 
# services:
#   db:
#     image: mariadb
#     container_name: nextcloud-mariadb
#     volumes:
#       - $HOME/pydoxtools/db:/var/lib/mysql
#       - /etc/localtime:/etc/localtime:ro
#     environment:
#       - MARIADB_ROOT_PASSWORD=y3pl5vb6qesu7vfbdtyntuh2fm093q206pk
#       - MARIADB_PASSWORD=haekf5ln0i4uy6hezn1wyvwjqy1qu4htx7n
#       - MARIADB_DATABASE=pydoxtools
#       - MARIADB_USER=pydoxtools
#     restart: unless-stopped
#     ports:
#       - 3808:3808
# ```

# %% tags=[]
def constraints(trial: optuna.Trial):
    return trial.user_attrs["constraint"]


sampler = optuna.samplers.NSGAIISampler(constraints_func=constraints)
study = optuna.create_study(
    study_name=study_name,
    storage=optuna.storages.RDBStorage(
        url=remote_storage,
        # engine_kwargs={"pool_size": 20, "connect_args": {"timeout": 10}},
        # add heartbeat i order to automatically mark "crashed" trials
        # as "failed" so that they can be repeated
        heartbeat_interval=60 * 5,
        grace_period=60 * 21,
    ),
    sampler=sampler,
    load_if_exists=True,
    directions=["maximize", "minimize"]
)

# %% tags=[]
"""
study.enqueue_trial(dict(
    rand_str_perc=0.1,
    random_char_prob=0.01,
    random_word_prob=0.01,
    random_upper_prob=0.01,
    mixed_blocks_generation_prob=0.1
))"""
if optimize:
    if len(sys.argv) > 1:
        study.optimize(train_model, n_jobs=int(sys.argv[1]), n_trials=1000)
    else:
        study.optimize(train_model, n_jobs=1, n_trials=1000)
else:
    train_model(optuna.trial.FixedTrial(dict(
        embeddings_dim=10,
        # embeddings vector size (standard BERT has a vector size of 768 )
        token_seq_length1=3,
        # what length of a word do we assume in terms of tokens?
        seq_features1=99,
        # how many filters should we run for the analysis = num of generated features?
        dropout1=0.5,  # first layer dropout
        cv_layers=1,  # number of cv layers
        token_seq_length2=74,
        # how many tokens in a row do we want to analyze?
        seq_features2=10,  # how many filters should we run for the analysis?
        dropout2=0.5,  # second layer dropout
        meanmax_pooling=0,  # whether to use meanmax_pooling at the end
        fft_pooling=1,  # whether to use fft_pooling at the end
        fft_pool_size=13  # size of the fft_pooling method
    )))
