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
import datetime
# %% [markdown] tags=[]
# # Train the Textblock classifier
#
# import datetime
# %% tags=[]
import logging
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
    _, _, m = training.prepare_textblock_training()
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
wu.rclone_single_sync_models(method="bisync", hostname=hostname, token=token, syncpath=syncpath)

# %%
f"sqlite:///{str(settings.MODEL_DIR)}/study.sqlite"

# %% tags=[]
# %env TOKENIZERS_PARALLELISM=true
if True:
    class WebdavSyncCallback(pytorch_lightning.Callback):
        def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            print("""lightning: sync models with rclone!""")
            print(wu.rclone_single_sync_models(
                method="copy", hostname=hostname, token=token, syncpath=syncpath)[0]
                  )


    additional_callbacks = [
        WebdavSyncCallback(),
        # pytorch_lightning.callbacks.RichProgressBar()
    ]

    import warnings

    warnings.filterwarnings("ignore", ".*Your `IterableDataset` has `__len__` defined.*")


    def train_model(trial: optuna.Trial):
        # we introduce a report call back in order to stop optimization runs early
        class ReportCallback(pytorch_lightning.Callback):
            def on_train_epoch_end(self, trainer: pytorch_lightning.Trainer, pl_module: "pl.LightningModule") -> None:
                score = trainer.callback_metrics['address.f1-score']
                trial.report(score, trainer.current_epoch)

        additional_callbacks.append(ReportCallback())

        data_config = dict(
            generators=[
                ("address", training.BusinessAddressGenerator(
                    rand_str_perc=trial.suggest_float("rand_str_perc", 0.0, 1.0))
                 ),
                ("unknown", training.RandomTextBlockGenerator()),
                ("unknown", training.RandomListGenerator()),
            ],
            weights=[10, 8, 2],
            cache_size=5000,
            renew_num=500,
            random_char_prob=trial.suggest_float("random_char_prob", 0.0, 0.5),
            random_word_prob=trial.suggest_float("random_word_prob", 0.0, 0.5),
            random_upper_prob=trial.suggest_float("random_upper_prob", 0.0, 0.5),
            mixed_blocks_generation_prob=trial.suggest_float("mixed_blocks_generation_prob", 0.0, 0.5),
            mixed_blocks_label="unknown",
        )

        model_config = dict(
            learning_rate=0.005,
            embeddings_dim=2,  # embeddings vector size (standard BERT has a vector size of 768 )
            token_seq_length1=5,  # what length of a work do we assume in terms of tokens?
            seq_features1=40,  # how many filters should we run for the analysis = num of generated features?
            dropout1=0.5,  # first layer dropout
            token_seq_length2=40,  # how many tokens in a row do we want to analyze?
            seq_features2=100,  # how many filters should we run for the analysis?
            dropout2=0.5,  # second layer dropout
            hp_metric="address.f1-score"  # the metric to optimize for and should be logged...
        )

        if False:
            m = classifier.txt_block_classifier.load_from_checkpoint(
                settings.MODEL_DIR / "text_blockclassifier-epoch=06-address.f1-score=0.24.ckpt.ckpt")
        else:
            m = None

        epoch_config = dict(
            steps_per_epoch=500,
            log_every_n_steps=50,
            max_epochs=10
        )
        epoch_config1 = dict(
            steps_per_epoch=5,
            log_every_n_steps=1,
            max_epochs=4
        )
        trainer, model, _, _ = training.train_text_block_classifier(
            train_model=True,
            log_hparams=trial.params,
            old_model=m,
            num_workers=4,
            accelerator="auto", devices=1,
            # strategy="ddp_find_unused_parameters_false",
            # strategy="ddp",
            strategy=None,  # in case of running jupyter notebook
            callbacks=additional_callbacks,
            data_config=data_config,
            model_config=model_config,
            **epoch_config
        )

        # trainer.logger.log_hyperparams()

        return trainer.callback_metrics['address.f1-score']


    study = optuna.create_study(
        study_name="find_data_generation_parameters",
        storage=f"sqlite:///{str(settings.MODEL_DIR)}/study.sqlite",
        load_if_exists=True,
        direction=optuna.study.StudyDirection.MAXIMIZE
    )

    """
    study.enqueue_trial(dict(
        rand_str_perc=0.1,
        random_char_prob=0.01,
        random_word_prob=0.01,
        random_upper_prob=0.01,
        mixed_blocks_generation_prob=0.1
    ))"""
    study.optimize(train_model, n_trials=100)

# %%
