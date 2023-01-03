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

# %% [markdown] tags=[]
# # Train the Textblock classifier
#
# import datetime
# %% tags=[]
import datetime
import logging
import platform
import warnings

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
if True:
    _, _, m = training.prepare_textblock_training()
    res = m.predict(["""ex king ltd
    Springfield Gardens
    Queens
    N. Y 11413
    www.something.com
    """,
                     """
                     some stupid text that doesn't mean anything...
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
print("sync with cloud...")
wu.rclone_single_sync_models(method="bisync", hostname=hostname, token=token, syncpath=syncpath)

# %% tags=[]
# %env TOKENIZERS_PARALLELISM=true
if True:
    warnings.filterwarnings("ignore", ".*Your `IterableDataset` has `__len__` defined.*")


    class WebdavSyncCallback(pytorch_lightning.Callback):
        def on_train_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
            print("""lightning: sync models with rclone!""")
            print(wu.rclone_single_sync_models(
                method="copy", hostname=hostname, token=token, syncpath=syncpath)[0]
                  )


    checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(
        monitor='address.f1-score',  # or 'accuracy' or 'f1'
        mode='max', save_top_k=3,
        dirpath=settings.MODEL_STORE("text_block").parent,
        filename="text_blockclassifier-{epoch:02d}-{address.f1-score:.2f}.ckpt"
    )

    additional_callbacks = [
        WebdavSyncCallback(), checkpoint_callback
        # pytorch_lightning.callbacks.RichProgressBar()
    ]

    data_config = dict(
        generators={
            "address": ((10, training.BusinessAddressGenerator(rand_str_perc=0.7)),),
            "unknown": (
                (5, training.RandomTextBlockGenerator()),
                (5, training.RandomListGenerator()))
        },
        cache_size=5000,
        renew_num=500,
        random_char_prob=0.45,
        random_word_prob=0.1,
        random_upper_prob=0.3,
        mixed_blocks_generation_prob=0.0,
        mixed_blocks_label="unknown",
    )

    model_config = dict(
        learning_rate=0.0005,
        embeddings_dim=4,  # embeddings vector size (standard BERT has a vector size of 768 )
        token_seq_length1=5,  # what length of a work do we assume in terms of tokens?
        seq_features1=40,  # how many filters should we run for the analysis = num of generated features?
        dropout1=0.5,  # first layer dropout
        token_seq_length2=40,  # how many tokens in a row do we want to analyze?
        seq_features2=100,  # how many filters should we run for the analysis?
        dropout2=0.5  # second layer dropout
    )

    if True:
        m = classifier.txt_block_classifier.load_from_checkpoint(settings.MODEL_DIR / "text_blockclassifier.ckpt")
    else:
        m = None
    trainer, model, train_loader, validation_loader = training.train_text_block_classifier(
        train_model=False,
        old_model=m,
        num_workers=8,
        accelerator="auto", devices=1,
        # strategy="ddp_find_unused_parameters_false",
        # strategy="ddp",
        strategy=None,  # in case of running jupyter notebook
        callbacks=additional_callbacks,
        enable_checkpointing=True,
        steps_per_epoch=500,
        log_every_n_steps=50,
        max_epochs=-1,
        data_config=data_config,
        model_config=model_config
    )

# %%
tune_learning_rate = True
if tune_learning_rate:
    lr_finder = trainer.tuner.lr_find(model, train_loader, validation_loader)

    # print(lr_finder.results)
    # Pick point based on plot, or get suggestion
    new_lr = lr_finder.suggestion()
    new_lr

    # - 0.00025 for a new model
    # - 0.00229 for a pre-trained model

    # Plot with
    fig = lr_finder.plot(suggest=True)
    fig.show()

# %%
new_lr

# %% [markdown]
# learning rate "fresh" model:   0.000363078054770101  (in case of fft)

# %%
pytorch_lightning.utilities.memory.get_model_size_mb(model)

# %%
model.hparams
