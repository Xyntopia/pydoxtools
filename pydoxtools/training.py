import collections
import datetime
import functools
import logging
import pickle
import random
import re
import string
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
    pageClassifier, txt_block_classifier
from pydoxtools.settings import settings

logger = logging.getLogger(__name__)
memory = settings.get_memory_cache()

_asciichars = ''.join(sorted(set(chr(i) for i in range(32, 128)).union(string.printable)))


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


class RandomTextBlockGenerator(GeneratorMixin):
    def __init__(self):
        self._f = f = open(settings.TRAINING_DATA_DIR / "all_text.txt", "rb")
        self._max_size = f.seek(0, 2)  # get the end f the text file
        self._separators = {
            " ": 10, ",": 2, "\n": 4, ";": 2, " |": 1,
            ":": 1, "/": 1
        }

    def single(self, seed, mix_blocks=1):
        rand = random.Random(seed)
        # = μ = α /
        mean_text_len = 100 * 3.5  # ~500 words is one page, 4.5 is the average word length in english
        var = 3.75 * 100  # σ²=α/β², σ/μ=1/sqrt(α) => α = μ²/σ²
        alpha = (mean_text_len / var) ** 2
        # β = μ / α
        txt_len = int(random.gammavariate(alpha=alpha, beta=mean_text_len / alpha)) + 1
        r1 = rand.randrange(0, self._max_size - txt_len)
        r2 = rand.randrange(0, self._max_size - txt_len)
        s1 = self._f.seek(r1)
        txt1 = self._f.read(txt_len).decode('utf-8', errors='ignore')
        if mix_blocks > 1:  # TODO: implement this
            raise NotImplementedError
            s2 = self._f.seek(r2)
            txt2 = self._f.read(txt_len).decode('utf-8', errors='ignore')
            randtxt = [rand_seps().join(r) for r in
                       list_utils.random_chunks(txt.split(), random.gammavariate)]

            rand_seps = rand_chars(separators)

        # add random seperators:
        return txt1


class BusinessAddressGenerator(GeneratorMixin):
    """TODO: convert this into  aker provider?"""

    def __init__(self, rand_str_perc=0.3):
        import faker
        self._available_locales = faker.config.AVAILABLE_LOCALES
        try:
            # this in order to avoid warning: "UserWarning: fr_QC locale is deprecated. Please use fr_CA."
            self._available_locales.remove("fr_QC")
        except ValueError:
            pass
        # pre-initialize fakers for all languages for speed
        self._fakers = {locale: faker.Faker(locale) for locale in self._available_locales}
        self._rand = random.Random()
        # function which returns some random separation characters
        self._sep_chars = rand_chars({"\n": 24, ", ": 12, "; ": 6, " | ": 6, "·": 1})
        self._rand_letter_street_perc = rand_str_perc

    def rand_word(self, f):
        mean_word_len = 4.5  # ~500 words is one page, 4.5 is the average word length in english, min len+1
        var = 3  # σ²=α/β², σ/μ=1/sqrt(α) => α = μ²/σ²
        alpha = (mean_word_len / var) ** 2
        # β = μ / α
        wordlen = int(random.gammavariate(alpha=alpha, beta=mean_word_len / alpha)) + 1
        return "".join(f.random_lowercase_letter() for i in range(wordlen))

    @cached_property
    def url_replace_regex(self):
        return re.compile(r'[^A-Za-z0-9-_]')

    def generate_url(self, company_name: str, tld: str) -> str:
        # generate a random url
        # remove "gmbh etc..  from company name
        company_name = company_name.lower()
        if self._rand.random() < 0.8:
            for i in ["gmbh", "& co. kg", "& co. ohg", "ohg", "inc", "ltd", "llc", "kgaa", "e.v.", "plc", "kg"]:
                company_name = company_name.replace(i, " ")

        safe_company_name = self.url_replace_regex.sub(self._rand.choice([r'-', '']), company_name).strip("-")
        domain_name = f"{safe_company_name}.{tld}"
        # replace double-dashes
        domain_name = re.sub(r'-+', '-', domain_name)
        url: str = self._rand.choice(["https://", "http://", "", ""]) \
                   + self._rand.choice(["www.", ""]) \
                   + domain_name

        r = self._rand.random()
        if r < 0.7:
            url = url.lower()
        elif r < 0.8:
            url = url.upper()
        # TODO: randomly write words with upper cases, trow in some upper case letters etc...
        # TODO: optionally add a path like "welcome" or "index" etc...
        # TODO:  add some variations like "de" as a language selection instead of "www."
        # TODO: remove "legal" company names from domain such as "GmbH",
        # TODO: add words different from company name...
        return url

    def random_streetname(self, f):
        """generate an address which consists
        - of purely random letter and numbers (resemble the structure of a real address)
        - or an augmented faker address
        """
        rc = self._rand.choice
        street_name = [self.rand_word(f) for i in range(self._rand.randint(1, 3))]
        if self._rand.random() > 0.1:
            street_name = street_name + [f.street_suffix()]
        street_addr = f.random_element(collections.OrderedDict(
            ((" ", 0.9), ("-", 0.05), ("", 0.05)))).join(street_name)
        building_num = str(self._rand.randint(0, 10000))
        return " ".join([building_num, street_addr][::rc((1, -1))])

    def random_city(self, f, s2, s3):
        city = []
        r = self._rand.random
        if r() > 0.5:  # prefix
            city += [self.rand_word(f)]
        # name
        city += [self.rand_word(f)]
        if r() > 0.5:  # suffix
            city += [self.rand_word(f)]

        city = " ".join(city)

        if r() < 0.8:  # state abbrev.
            state = "".join(f.random_uppercase_letter() for i in range(self._rand.randint(2, 3)))
        else:  # state
            state = self.rand_word(f)

        postcode = str(self._rand.randint(1000, 99999))

        selec = r()
        if selec < 0.5:
            return f"{city}{s2} {state} {s3} {postcode}"
        else:
            return f"{postcode} {city}{s2} {state}"

    def single(self, seed) -> str:
        import faker
        faker.Faker.seed(seed)
        self._rand.seed(seed)
        rc, r = self._rand.choice, self._rand.random
        f = self._fakers[rc(self._available_locales)]
        line_separator = (" " if r() > 0.9 else "") + self._sep_chars() + (" " if r() > 0.5 else "")
        # next level separators
        s2_choices = {",", ";", ",", "|", "·"} - {line_separator}
        s2, s3 = f.random_elements(elements=s2_choices, length=2, unique=True)

        company: str = f.company()

        faker_address = None

        def get_faker_address():
            nonlocal faker_address
            if faker_address:
                return faker_address
            else:
                faker_address = f.address().split("\n")
                return faker_address

        def addr1():
            if random.random() < self._rand_letter_street_perc:
                return self.random_streetname(f)
            else:
                return get_faker_address()[0]

        def addr3():
            if random.random() < self._rand_letter_street_perc:
                return self.random_city(f, s2, s3)
            else:
                try:
                    return get_faker_address()[1]
                except IndexError:
                    return " "

        def get_country():
            try:
                selector = r()
                if selector < 0.5:
                    return f.current_country_code()
                elif selector < 0.9:
                    return f.current_country()
                else:
                    return self.rand_word(f)
            except (ValueError, AttributeError):
                return " "

        def line1() -> str:
            """introducing the address"""
            if r() > 0.5:
                line = f.catch_phrase()
            else:
                line = self.rand_word(f)

            line += rc((":", ""))
            line += "\n" * random.randint(0, 5)
            return line

        all_parts = {
            ("",): (0.05, line1),  # for example an announcement, or company introduction, catch phrase...
            ("name",): (0.1, lambda: f.name()),
            ("company_name", "company"): (0.3, lambda: company),
            ("address", "mailing address", "mail address"): (0.9, addr1),  # street
            ("",): (
                0.05, lambda: s2.join([self.rand_word(f), str(self._rand.randint(0, 1000))][::rc((1, -1))])),
            # postbox, building number etc...
            ("",): (0.9, addr3),  # city, postcode, state
            ("country",): (0.1, get_country),
            ("tel", "phone", "phone number"): (0.2, lambda: getattr(f, "phone_number", lambda: " ")()),
            ("fax", "fax number"): (0.1, lambda: getattr(f, "phone_number", lambda: " ")()),
            ("www", "web", "homepage", "webpage"): (0.2, lambda: self.generate_url(company, f.tld())),
            ("email", ""): (0.1, lambda: f.email()),
            ("bank details"): (0.05, lambda: " ".join(self.rand_word(f) for i in range(self._rand.randint(1, 5))))
        }

        # select random selection of address parts
        parts = set()
        while len(parts) < 3:  # make sure addresses consist of a min-length
            for k, (p, v) in all_parts.items():
                if k not in parts:
                    if r() < p:
                        parts.add(k)

        # TODO: augment address with "labels" such as "address:"  and "city" and
        #       "Postleitzahl", "country" etc...
        #       we can use a callable like this:
        #       def rand_func(match_obj):  and return a string based on match_obj

        # render the actual address
        render_parts = []
        for k, (p, v) in all_parts.items():
            if k in parts:
                try:
                    v = v()
                except:
                    logger.exception("error in address generation function")
                    v = " "
                if k[0]:
                    if r() < 0.2:
                        field_name = rc(k) if r() < 0.7 else self.rand_word(f)
                        v = field_name + rc((" ", ": ", ":")) + v
                if r() < 0.3:  # probability to have a part in upper case
                    v = v.upper()
                elif r() < 0.5:  # only first letter
                    v = v[0].upper() + v[1:]
                else:  # everything lower case
                    v = v.lower()

                render_parts.append(v)

        # randomly switch certain address lines for ~10%
        if r() < 0.1:
            a, b = random.randrange(0, len(render_parts)), random.randrange(0, len(render_parts))
            render_parts[a], render_parts[b] = render_parts[b], render_parts[a]

        # TODO: make a variation of separators for phone numbers...
        # TODO: discover other ways that addresses for use in our generators...
        address = line_separator.join(render_parts)
        return address


class RandomListGenerator(GeneratorMixin):
    # faker.pylist
    def __init__(self):
        import faker
        self._faker = f = faker.Faker()
        self._fake_methods = [m for m in [m for m in dir(f) if not m == "seed"]
                              if callable(getattr(f, m)) and m[0] != "_"]

    def single(self, seed):
        import faker
        faker.Faker.seed(seed)
        rand = random.Random(seed)
        f = self._faker

        # define list symbol
        list_indicators = [""] + list(" -*∙•+~")
        list_symbol = rand.choice(list_indicators)
        # define symbols to choose from
        a = '??????????###############--|.:?/\\'
        strlen = min(int(random.gammavariate(1.7, 3.8)) + 1, len(a))
        frmtstr = "".join(rand.sample(a, strlen))
        list_word = list(f.pystr_format(string_format=frmtstr))
        # replace a random number of letters with numbers/letters
        for i in range(rand.randint(0, len(list_word) // 2)):
            list_word[rand.randint(0, len(list_word)) - 1] = "#" if rand.random() > 0.2 else "?"
        list_word = "".join(list_word)

        # TODO: add more sophisticated lists..  with words
        #       - only with numbers
        #       - with the same word over & over
        #       - with the same symbols over & over
        # TODO: add lists with values from "self._fake_methods"
        # datalen=min(int(random.weibullvariate(8,1.4))+1,len(fake_methods))
        # frmtstr="{{"+'}}{{'.join(random.sample(fake_methods, datalen))+"}}"
        # f.pystr_format(string_format = frmtstr), frmtstr)

        prefix = list_symbol + random.choice(["", " "])
        res = []
        for i in range(rand.randint(2, 30)):
            res.append(prefix + f.pystr_format(string_format=list_word))

        separator = f.random_element(collections.OrderedDict(
            (("\n", 0.4), (", ", 0.2), (",", 0.2), (" ", 0.2))
        ))

        return separator.join(res)


class RandomTableGenerator(GeneratorMixin):
    # faker.dsv
    pass


class RandomDataGenerator(GeneratorMixin):
    # TODO: faker.pydict
    #       then convert to json, yaml, xml etc...
    pass


class RandomCodeGenerator():
    # TODO: build some random code genrator in some "iterative" way
    #       maybe downloading some langauge grammar?
    #       definitly using language keywords...
    #       possibly making use of github datasets...
    pass


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


class PandasDataframeLoader(torch.utils.data.Dataset):
    def __init__(self, X: pd.DataFrame, y: pd.Series):
        self._X = X
        self._y = y

    def __getitem__(self, i):
        return self._X[i], self._y[i]

    def __len__(self):
        return len(self._X)


class TextBlockGenerator(torch.utils.data.IterableDataset):
    def __init__(
            self,
            generators: list[tuple[str, typing.Any]],
            weights: list[int],
            random_char_prob: float = 0.0,
            random_word_prob: float = 0.0,
            random_upper_prob: float = 0.0,
            mixed_blocks_generation_prob: float = 0.0,
            mixed_blocks_label: str = "unknown",
            cache_size: int = 10000,
            renew_num: int = 1000,
            virtual_size: int = 100000
    ):
        """
        augment_prop: augmentation is mainly used if we are training making the random samples more iverse with
                        random letters etc....
        mixed_blocks_generation_prob:  append blocks from different generators to each other
                                       and give them a label according to "mixed_blocks_label"  option

        cache_size: Maximum cache size. The cache is used to speed up the random generator
                    by reusing old generated sample every loop
        renew_num: number of elements of the cache to be replaces on every iteration
        virtual_size: defines what the "__len__" gives back
        """
        self._generators = generators
        self._weights = weights
        self._random_char_prob = random_char_prob
        self._random_word_prob = random_word_prob
        self._random_upper_prob = random_upper_prob
        self._mixed_blocks_generation_prob = mixed_blocks_generation_prob
        self._mixed_blocks_label = mixed_blocks_label
        self._random = random.Random(time.time())  # get our own random number generator
        self._cache_size = cache_size
        self._renew_num = renew_num
        self._virtual_size = virtual_size
        import faker
        self._available_locales = faker.config.AVAILABLE_LOCALES
        try:
            # this in order to avoid warning: "UserWarning: fr_QC locale is deprecated. Please use fr_CA."
            self._available_locales.remove("fr_QC")
        except ValueError:
            pass
        # pre-initialize fakers for all languages
        self._faker = faker.Faker(self._available_locales)

    @cached_property
    def class_gen(self):
        return list(j for _, j in self._generators)

    @cached_property
    def num_generators(self):
        return len(self._generators)

    @cached_property
    def classmap(self):
        classes = set(i for i, _ in self._generators)
        return dict(enumerate(classes))

    @cached_property
    def classmap_inv(self):
        return {j: i for i, j in self.classmap.items()}

    def single(self, seed):
        # TODO: does it really make sense here to keep seed?
        #       only makes sense if EVERY random choice is 100% included in it
        self._random.seed(seed)

        # randomly choose a generator for a certain class
        cl, gen = self._random.choices(population=self._generators, weights=self._weights)[0]
        # generate random input data for the class to be predicted
        x = gen(seed)
        if self._mixed_blocks_generation_prob > self._random.random():
            cl, gen = self._random.choices(population=self._generators, weights=self._weights)[0]
            x += "\n" + gen(seed + 1)
            y = self.classmap_inv[self._mixed_blocks_label]
        else:
            y = self.classmap_inv[cl]

        # word augmentation
        if self._random_word_prob:
            # we use some augmentation for our dataset here to make the classifier more robust...
            t = []
            # we would like to preserve newline chars thats why we are splitting in such a "funny" way
            words = x.replace("\n", " \n ").split(" ")
            wordnum = len(words)
            for i, w in enumerate(words):
                # TODO:  merge word and char augmentation into a general "list-augmenation" function
                if w and (self._random.random() < self._random_word_prob):
                    selector = self._random.randint(0, 10)
                    if selector == 0:  # substitute with random char sequences
                        t.append(self._faker.bothify(text="?" * self._random.randint(1, 15)))
                    elif selector == 1:  # insert before
                        t.append(self._faker.bothify(text="?" * self._random.randint(1, 15)))
                        t.append(w)
                    elif selector == 2:  # insert after
                        t.append(w)
                        t.append(self._faker.bothify(text="?" * self._random.randint(1, 15)))
                    elif selector == 3:  # delete
                        continue
                    elif selector == 4:  # duplicate
                        t.append(w)
                        t.append(w)
                    elif selector == 5:  # swap
                        i2 = self._random.randrange(0, wordnum)
                        t.append(words[i2])
                        words[i2] = w
                    elif selector == 6:  # neighbourswap
                        if self._random.random() > 0.5:
                            i2 = min(i + 1, wordnum - 1)
                        else:
                            i2 = max(i - 1, 0)
                        t.append(words[i2])
                        words[i2] = w
                    elif selector == 7:  # one-side-swap (only substitue the item with a duplicate from somewhere else in the string)
                        t.append(self._random.choice(words))
                    # already handled by inserting random characters
                    # elif selector ==8: #randomly split words
                    #    splitnum=self._random.randrange(0,len(w))
                    #    t.extend([w[:splitnum],w[splitnum:]])
                    elif selector == 8:  # reverse
                        t.append(w[::-1])
                    elif selector == 9:  # crop
                        crop = self._random.randrange(len(w))
                        if crop > (len(w) // 2):
                            t.append(w[crop:])
                        else:
                            t.append(w[:crop])
                    elif selector == 10:  # convert to upper case
                        if self._random.random() > 0.5:
                            t.append(w.upper())
                        elif len(w) > 1:  # only first letters
                            t.append(w[0].upper() + w[1:])
                else:
                    t.append(w)
            x = " ".join(t).replace(" \n ", "\n")

        # character augmentation
        if self._random_char_prob:
            # we use some augmentation for our dataset here to make the classifier more robust...
            t = []
            x_list = list(x)
            lenx = len(x)
            for i, c in enumerate(x_list):
                if self._random.random() < self._random_char_prob:
                    selector = self._random.randint(0, 8)
                    if selector == 0:  # substitute
                        t.append(self._random.choice(_asciichars))
                    elif selector == 1:  # insert before
                        t.append(self._random.choice(_asciichars))
                        t.append(c)
                    elif selector == 2:  # insert after
                        t.append(c)
                        t.append(self._random.choice(_asciichars))
                    elif selector == 3:  # delete
                        continue
                    elif selector == 4:  # duplicate
                        t.append(c)
                        t.append(c)
                    elif selector == 5:  # swap
                        i2 = self._random.randrange(0, lenx)
                        t.append(x_list[i2])
                        x_list[i2] = c
                    elif selector == 6:  # neighbourswap
                        if self._random.random() > 0.5:
                            i2 = min(i + 1, lenx - 1)
                        else:
                            i2 = max(i - 1, 0)
                        t.append(x_list[i2])
                        x_list[i2] = c
                    elif selector == 7:  # one-side-swap (only substitue the item with a duplicate from somewhere else in the string)
                        t.append(self._random.choice(x))
                    elif selector == 8:
                        t.append(c.upper())
                else:
                    t.append(c)

            x = "".join(t)

        # whole text augmentation
        if self._random.random() < self._random_upper_prob:
            selec = self._random.random()
            if selec < 0.45:
                x = x.upper()
            elif selec < 0.5:
                x = "".join(
                    " ".join([w[0].upper() + w[1:].lower()]) for w in x.replace("\n", " \n ").split(" ")
                    if len(w) > 0)
            elif selec < 0.55:
                x = "".join(" ".join([w[0].upper() + w[1:]]) for w in x.replace("\n", " \n ").split(" ")
                            if len(w) > 0)
            else:
                x = x.lower()

        return x, torch.tensor(y)

    def __iter__(self):
        # initialize cache with random samples
        cache = collections.deque(
            (self.single(
                seed=self._random.random()) for i in range(self._cache_size)),
            maxlen=self._cache_size)
        while True:
            # serve all sample currently in cache
            yield from cache
            # add new samples to the cache collections deque automatically drops old samples
            # that exceed _cache_size
            cache.extend(self.single(seed=self._random.random()) for i in range(self._renew_num))

    def __len__(self):
        return self._virtual_size


def prepare_textblock_training(
        num_workers: int = 4,
        steps_per_epoch: int = 500,
        batch_size=2 ** 10,
        data_config=None,
        model_config=None
):
    """
    loads text block data and puts into pytorch dataloaders

    # TODO: this is the right place for hyperparameteroptimization....

    TODO: we also need to detect the pdf langauge in order
          to have a balances dataset with addresses from different countries
    """
    # TODO: build a "validation" loader with df_labeled...
    # label_file = settings.TRAINING_DATA_DIR / "labeled_txt_boxes.xlsx"
    # df_labeled = pd.read_excel(label_file)

    virtual_size = steps_per_epoch * batch_size
    user_generator_config = dict(
        generators=[
            ("address", BusinessAddressGenerator()),
            ("unknown", RandomTextBlockGenerator()),
            ("unknown", RandomListGenerator()),
        ],
        weights=[10, 8, 2],
        random_char_prob=1.0 / 20.0,
        random_word_prob=1.0 / 20.0,
        cache_size=5000,
        renew_num=500,
        mixed_blocks_generation_prob=0.1,
        mixed_blocks_label="unknown",
        virtual_size=virtual_size
    )
    user_generator_config.update(data_config or {})
    train_dataset = TextBlockGenerator(**user_generator_config)

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
        multiaddress="unknown"
    ))
    y = y.replace(train_dataset.classmap_inv)
    assert y.dtype == np.dtype('int64'), f"has to be numeric!! y.unique: {y.unique()}"
    test_dataset = PandasDataframeLoader(X=df['txt'], y=y)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        persistent_workers=True
        # sampler=weighted_sampler
    )
    validation_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=len(test_dataset),
        num_workers=num_workers,
        persistent_workers=True
        # sampler=weighted_sampler
    )

    classmap = train_dataset.classmap
    logger.info(f"extracted classes: {classmap}")
    model = txt_block_classifier(classmap, **(model_config or {}))

    return train_loader, validation_loader, model


@functools.lru_cache()
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
        log_hparams: dict = None,
        old_model=None,
        num_workers=4,
        steps_per_epoch=200,
        data_config=None,
        model_config=None,
        **kwargs,
):
    train_loader, validation_loader, model = prepare_textblock_training(
        num_workers, steps_per_epoch=steps_per_epoch, data_config=data_config, model_config=model_config
    )
    if old_model:
        model = old_model
    trainer = pytorch_lightning.Trainer(
        accelerator=kwargs.get('accelerator', 'cpu'),
        devices=kwargs.get('devices', 1),
        strategy=kwargs.get('strategy', 'ddp'),
        # gpus=-1, auto_select_gpus=True,s
        log_every_n_steps=kwargs.get("log_every_n_steps", 100),
        # limit_train_batches=100,
        max_epochs=kwargs.get("max_epochs", -1),
        # checkpoint_callback=False,
        enable_checkpointing=kwargs.get("enable_checkpointing", False),
        max_steps=-1,
        # auto_scale_batch_size=True,
        callbacks=kwargs.get('callbacks', []),
        default_root_dir=settings.MODEL_STORE("text_block").parent
    )
    if kwargs.get("train_model", False):
        # TODO: ho can we set hp_metric to 0 at the start?
        trainer.logger.log_hyperparams(log_hparams)  # , metrics=dict(hp_metric=0))
        trainer.fit(model, train_loader, validation_loader)
        # trainer.logger.log_hyperparams(log_hparams, metrics=dict(
        #    model.metrics
        # ))
        trainer.save_checkpoint(settings.MODEL_STORE("text_block"))
        curtime = datetime.datetime.now()
        curtimestr = curtime.strftime("%Y%m%d%H%M")
        trainer.save_checkpoint(
            settings.MODEL_STORE(
                "text_block").parent / f"text_blockclassifier_v{trainer.logger.version}_{curtimestr}.ckpt")

    return trainer, model, train_loader, validation_loader


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
