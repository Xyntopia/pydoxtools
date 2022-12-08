import datetime
import datetime
import functools
import logging
import pickle
import random
import re
import time
import typing
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd
import pytorch_lightning
import sklearn
import torch

from pydoxtools import file_utils, list_utils
from pydoxtools.classifier import pdfClassifier, gen_meta_info_cached, \
    pageClassifier, _asciichars, txt_block_classifier
from pydoxtools.settings import settings

logger = logging.getLogger(__name__)
memory = settings.get_memory_cache()


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


class GeneratorMixin:
    def __getitem__(self, ids: list[int] | int):
        return self(ids)

    def __call__(self, ids: list[int] | int) -> str | list[str]:
        # TODO: international addresses
        # TODO: more variations
        if isinstance(ids, list):
            return [self.single(id) for id in ids]
        else:
            return self.single(ids)


def weibull_ints(k=1.4, l=8, rand=None):
    # k = scale, l = shape
    # this function helps us o generate a "realistic" distribution of textblock sizes
    if rand:
        return int(rand.weibullvariate(l, k))
    else:
        return int(random.weibullvariate(l, k))


class RandomTextBlockGenerator(GeneratorMixin):
    def __init__(self):
        self._f = f = open("/home/tom/comcharax/data/raw_text/all_text.txt", "rb")
        self._max_size = f.seek(0, 2)
        self._separators = {
            " ": 10, ",": 2, "\n": 4, ";": 2, " |": 1,
            ":": 1, "/": 1
        }

    def single(self, id, mix_blocks=1):
        rand = random.Random(id)
        txt_len = weibull_ints(l=100, rand=rand)
        r1 = rand.randrange(0, self._max_size - txt_len)
        r2 = rand.randrange(0, self._max_size - txt_len)
        s1 = self._f.seek(r1)
        txt1 = self._f.read(txt_len).decode('utf-8', errors='ignore')
        if mix_blocks > 1:  # TODO: imple,ent this
            raise NotImplementedError
            s2 = self._f.seek(r2)
            txt2 = self._f.read(txt_len).decode('utf-8', errors='ignore')
            randtxt = [rand_seps().join(r) for r in
                       list_utils.random_chunks(txt.split(), weibull_ints)]

            rand_seps = rand_chars(separators)

        # add random seperators:
        return txt1


class BusinessAddressGenerator(GeneratorMixin):
    def __init__(self, fake_langs):
        """fake_langs can be something like:
        ['en_US', 'de_DE', 'en_GB']...
        """
        from faker import Faker
        import international_address_formatter
        self._fake_langs = fake_langs or ['en_US', 'de_DE', 'en_GB']
        self._countrycodes = {i: i[-2:] for i in self._fake_langs}
        self._Faker = Faker
        self._IAF = international_address_formatter
        self._faker = Faker(self._fake_langs)

    @cached_property
    def url_replace_regex(self):
        return re.compile(r'[^A-Za-z0-9-_]')

    def generate_url(self, rand: random.Random, company_name: str, tld: str) -> str:
        # generate a random url
        # remove "gmbh etc..  from company name
        company_name = company_name.lower()
        if rand.random() < 0.8:
            for i in ["gmbh", "& co. kg", "& co. ohg", "ohg", "inc", "ltd", "llc", "kgaa", "e.v.", "plc", "kg"]:
                company_name = company_name.replace(i, " ")

        safe_company_name = self.url_replace_regex.sub(rand.choice([r'-', '']), company_name).strip("-")
        domain_name = f"{safe_company_name}.{tld}"
        # replace double-dashes
        domain_name = re.sub(r'-+', '-', domain_name)
        url: str = rand.choice(["https://", "http://", "", ""]) \
                   + rand.choice(["www.", ""]) \
                   + domain_name

        r = rand.random()
        if r < 0.7:
            url = url.lower()
        elif r < 0.8:
            url = url.upper()
        # TODO: randomly write words with upper cases, trow in some upper case letters etc...
        # TODO: optionally add a path like "welcome" or "index" etc...
        # TODO:  add some variations like "de" as a language selection instead of "www."
        # TODO: remove "legal" company names from domain such as "GmbH",
        return url

    def single(self, id: int) -> str:
        self._Faker.seed(id)
        rand = random.Random(id)
        rc, r = rand.choice, rand.random
        locale = rc(self._fake_langs)
        country_code = self._countrycodes[locale]
        fake = self._faker[locale]
        company: str = fake.company()
        phone = rc(["", "Phone ", "Phone: "]) + fake.phone_number()
        fax = rc(["", "fax ", "fax: "]) + fake.phone_number()
        formatter = self._IAF.AddressFormatter("DE")

        # TODO: use better address genereation, making use of osmand data
        if r() < 0.3:
            fax = fax.upper()
            phone = phone.upper()
        parts = dict(
            name=company if r() > 0.4 else company.upper(),
            address=fake.address(),
            phone=phone,
            fax=fax,
            www=self.generate_url(rand, company, fake.tld())
        )
        # leave out every n_th line by random chance
        sep = rc(["\n", "; ", ", "])
        # randomly switch certain address lines for ~10%
        parts = list(parts.values())
        if r() < 0.1:
            a, b = random.randrange(0, len(parts)), random.randrange(0, len(parts))
            parts[a], parts[b] = parts[b], parts[a]

        address = sep.join(line for line in parts if r() > 0.3)
        return address


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


def random_string_augmenter(data: str, prob: float) -> str:
    return "".join(c if random.random() > prob else random.choice(_asciichars) for c in data)


class TextBlockGenerator(torch.utils.data.IterableDataset):
    def __init__(self, generators: dict[str, typing.Any], augment_prob: int = 0.0):
        """augmentation is mainly used if we are training..."""
        self._generators = generators
        self.augment = augment_prob
        self.random = random.Random(time.time())  # get our own random number generator

    @cached_property
    def class_gen(self):
        return list(self._generators.values())

    @cached_property
    def num_generators(self):
        return len(self._generators)

    @cached_property
    def classmap(self):
        return dict(enumerate(self._generators.keys()))

    def single(self, i):
        # TODO: introduce some caching mechanisms..   provide already cached data over time and
        #       introduce new samples at a 1% rate or smoething like that to improve speed...
        # we don't need to return x as a "list", as our textblock model accepts a list of strings directly
        # randomly choose a generator for a certain class
        y = self.random.randint(0, self.num_generators - 1)
        # generate random input data for the class to be predicted
        x = self.class_gen[y](i)

        if self.augment:
            # we use some augmentation for our dataset here to make the classifier more robust...
            return random_string_augmenter(x, prob=self.augment), torch.tensor(y)
        else:
            return x, torch.tensor(y)

    def __iter__(self):
        for i in range(100000000000000):
            yield self.single(i)


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
    # TODO: replace this "labeled" xt blocks iwth metadata from TextBlockGenerator initialization
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
    train_dataset = TextBlockGenerator(generators={}, augment_prob=1.0 / 50.0)
    test_dataset = TextBlockGenerator(generators={}, )
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


def train_text_block_classifier(old_model=None, num_workers=4, **kwargs):
    train_loader, test_loader, model = prepare_textblock_training(num_workers)
    if old_model:
        model = old_model
    checkpoint_callback = pytorch_lightning.callbacks.ModelCheckpoint(
        monitor='train_loss',  # or 'accuracy' or 'f1'
        mode='min', save_top_k=5,
        dirpath=settings.MODEL_STORE("text_block").parent,
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
        default_root_dir=settings.MODEL_STORE("text_block").parent
    )
    if True:
        # TODO: normally this should be discouraged to do this...
        #       (using test dataset for validation) ...
        #       but we don't have enough test data yet.
        val_dataloader = test_loader
        trainer.fit(model, train_loader, val_dataloader)
        trainer.save_checkpoint(settings.MODEL_STORE("text_block"))
        curtime = datetime.datetime.now()
        curtimestr = curtime.strftime("%Y%m%d%H%M")
        trainer.save_checkpoint(
            settings.MODEL_STORE("text_block").parent / f"text_blockclassifier{curtimestr}.ckpt")

    return trainer.test(model, test_dataloaders=test_loader, ckpt_path=settings.MODEL_STORE('text_block')), model


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
