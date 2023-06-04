from __future__ import annotations

import abc
import collections
import pathlib
import random
import re
import string
import time
import typing
from functools import cached_property
from itertools import islice
from random import randint

import torch

from .settings import settings

_asciichars = ''.join(sorted({chr(i) for i in range(32, 128)}.union(string.printable)))


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


def rand_exec(r: random.random, func: typing.Callable, prob: float):
    """randomly executes a function or returns empty string"""
    if r() < prob:
        return func()
    else:
        return ""


class GeneratorMixin:
    def __getitem__(self, ids: list[int] | int):
        return self(ids)

    @abc.abstractmethod
    def single(self, seed):
        pass

    def __call__(self, ids: list[int] | int) -> str | list[str]:
        # TODO: international addresses
        # TODO: more variations
        if isinstance(ids, list):
            return [self.single(id) for id in ids]
        else:
            return self.single(ids)


class TextResource:
    def __init__(self, filename: str | pathlib.Path):
        self._f = f = open(filename, "rb")
        self._max_size = f.seek(0, 2)  # get the end f the text file

    def random_str(self, str_len: int) -> str:
        pos = random.randrange(0, self._max_size - str_len)
        self._f.seek(pos)
        txt = self._f.read(str_len).decode('utf-8', errors='ignore')
        return txt

    def random_line(self, max_search: int = 100) -> str:
        """returns a random line it will search for preceding newline character maximum of "max_search" characters"""
        pos = random.randrange(0, self._max_size)
        if pos < max_search:
            max_search2 = max(pos - max_search, pos)
        else:
            max_search2 = max_search
        self._f.seek(pos - max_search2)  # only search in the 100 lines before cursor position
        nl_idx = self._f.read(max_search2).rfind(b"\n")
        self._f.seek(pos - max_search2 + nl_idx + 1)
        txt = self._f.readline().decode('utf-8', errors='ignore').rstrip()
        return txt


class RandomTextBlockGenerator(GeneratorMixin):
    def __init__(self, txt_source: pathlib.Path):
        self._txt_res = TextResource(txt_source)
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
        txt1 = self._txt_res.random_str(txt_len)
        if mix_blocks > 1:  # TODO: implement this, combine two different text strings into one
            raise NotImplementedError
            s2 = self._f.seek(r2)
            txt2 = self._f.read(txt_len).decode('utf-8', errors='ignore')
            randtxt = [rand_seps().join(r) for r in
                       random_chunks(txt.split(), random.gammavariate)]

            rand_seps = rand_chars(separators)

        # add random seperators:
        return txt1


class BusinessAddressGenerator(GeneratorMixin):
    """TODO: convert this into  Faker provider?"""

    def __init__(self, type="address", rand_str_perc=0.3, osm_perc=0.5, fieldname_prob=0.05):
        # function which returns some random separation characters
        self._rand_str_perc = rand_str_perc  # probability to use radom strings in certain areas
        self._osm_perc = osm_perc  # probability to use openstreetmap data
        self._fieldname_prob = fieldname_prob  # probability to use a fieldname in front of an address line
        self._f: faker.Faker | None = None
        self._type: str = type

    @cached_property
    def random_sep_func(self):
        return rand_chars({"\n": 48, ", ": 12, "; ": 6, " | ": 3, "|": 3, "·": 1, " · ": 1})

    @cached_property
    def _rand(self):
        return random.Random()

    @cached_property
    def _available_locales(self):
        import faker
        available_locales = faker.config.AVAILABLE_LOCALES
        try:
            # this in order to avoid warning: "UserWarning: fr_QC locale is deprecated. Please use fr_CA."
            available_locales.remove("fr_QC")
        except ValueError:
            pass
        return available_locales

    @cached_property
    def _fakers(self):
        import faker
        # pre-initialize fakers for all languages for speed
        fakers: dict[str, faker.Faker] = {locale: faker.Faker(locale) for locale in self._available_locales}
        return fakers

    @cached_property
    def osm_resource(self) -> dict[str, TextResource]:
        osm_data_files = dict(
            cities="cities.txt",
            streets="streets.txt",
            names="names.txt",
            states="states.txt"
        )
        return {
            k: TextResource(settings.TRAINING_DATA_DIR / v) for k, v in osm_data_files.items()
        }

    def reset_faker(self, seed=None):
        if seed:
            import faker
            faker.Faker.seed(seed)
            self._rand.seed(seed)
        rc, r = self._rand.choice, self._rand.random
        # create a bias towards US, GB and DE addresses, as they represent the
        # majority in our dataset
        rv = r()
        if rv < 0.3:
            flocal = rc(self._available_locales)
        elif rv < 0.5:
            flocal = "de_DE"
        elif rv < 0.7:
            flocal = "en_GB"
        else:
            flocal = "en_US"

        self._f = self._fakers[flocal]

    def rand_word(self):
        mean_word_len = 4.5  # ~500 words is one page, 4.5 is the average word length in english, min len+1
        var = 3  # σ²=α/β², σ/μ=1/sqrt(α) => α = μ²/σ²
        alpha = (mean_word_len / var) ** 2
        # β = μ / α
        wordlen = int(random.gammavariate(alpha=alpha, beta=mean_word_len / alpha)) + 1
        return "".join(self._f.random_lowercase_letter() for i in range(wordlen))

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

    def random_streetname(self):
        """generate an address which consists
        - of purely random letter and numbers (resemble the structure of a real address)
        """
        f = self._f
        street_name = [self.rand_word() for i in range(self._rand.randint(1, 3))]
        if self._rand.random() < 0.9:  # add a suffix
            street_name = street_name + [f.street_suffix()]
        # add random hyphenation
        street_name = f.random_element(collections.OrderedDict(
            ((" ", 0.9), ("-", 0.05), ("", 0.05)))).join(street_name)
        return street_name

    def random_city_name(self):
        # generate completly random city
        city = []
        r = self._rand.random
        if r() > 0.5:  # prefix
            city += [self.rand_word()]
        # name
        city += [self.rand_word()]
        if r() > 0.5:  # suffix
            city += [self.rand_word()]

        city = " ".join(city)
        return city

    def random_state(self):
        r = self._rand.random
        if r() < 0.5:  # state abbrev.
            state = "".join(self._f.random_uppercase_letter() for i in range(self._rand.randint(2, 3)))
        else:  # state
            state = self.rand_word()
        return state

    def get_country(self):
        selector = self._rand.random()
        try:
            if selector < 0.5:
                return self._f.current_country_code()
            elif selector < 0.9:
                return self._f.current_country()
        except (ValueError, AttributeError):
            pass

        return self.rand_word()

    def addr_line3(self, s2, s3, city, state):
        r = self._rand.random
        postcode = str(self._rand.randint(1000, 99999))

        return self._rand.choice((
            f"{city}{s2} {state} {s3} {postcode}",
            f"{postcode} {city}{s2} {state}",
            f"{city} {s2} {postcode} {state}",
            f"{city} {postcode} {state}"
        ))

    def line1(self) -> str:
        """introducing the address"""
        if self._rand.random() < self._rand_str_perc:
            line = self.rand_word()
        else:
            line = self._f.catch_phrase()

        line += self._rand.choice((":", ""))
        line += "\n" * random.randint(0, 5)
        return line

    def single(self, seed) -> str:
        """
        TODO: we need to make this function vastly more efficient and more
              readable
        """
        self.reset_faker(seed)
        rc, r = self._rand.choice, self._rand.random
        line_separator = self.random_sep_func()  # calling our cached random seperator_generation function
        # next level separators
        s1 = line_separator.strip(" ")
        s2_choices = [s for s in (",", "; ", ";", ", ", "|", " | ", "·", " · ") if (s1 not in s)]
        s2, s3 = self._f.random_elements(elements=s2_choices, length=2, unique=True)

        company: str
        if r() < self._osm_perc:
            company = self.osm_resource["names"].random_line().strip('"')
        else:
            company = self._f.company()

        faker_address = None

        def get_faker_address():
            nonlocal faker_address
            if faker_address:
                return faker_address
            else:
                faker_address = self._f.address().split("\n")
                return faker_address

        def addr1():
            if r() < self._osm_perc:
                street_name = self.osm_resource['streets'].random_line().strip('"')
                building_num = str(self._rand.randint(0, 10000))
                return " ".join([building_num, street_name][::rc((1, -1))])
            else:
                if r() < self._rand_str_perc:
                    street_name = self.random_streetname()
                    building_num = str(self._rand.randint(0, 10000))
                    return " ".join([building_num, street_name][::rc((1, -1))])
                else:
                    return get_faker_address()[0]

        def addr3():
            if r() < self._osm_perc:
                city = self.osm_resource['cities'].random_line().strip('"')
                state = self.osm_resource['states'].random_line().strip('"')
                return self.addr_line3(s2=s2, s3=s3, city=city, state=state)
            else:
                if r() < self._rand_str_perc:
                    city = self.random_city_name()
                    state = self.random_state()
                    return self.addr_line3(s2=s2, s3=s3, city=city, state=state)
                else:
                    try:
                        return get_faker_address()[1]
                    except IndexError:
                        return ""

        # function definition:
        all_parts: dict[tuple[tuple[str, ...], typing.Callable]] = {
            "intro": (("",), lambda: self.line1()),
            # for example an announcement, or company introduction, catch phrase...
            "name": (("name", "Attn"),
                     lambda: rand_exec(r, lambda: self._f.job() + " ", 0.2) + self._f.name()),
            "company": (("company name", "company"), lambda: company),
            "space": (("",), lambda: "\n" * self._rand.randint(0, 5)),
            "street_address": (("add", "address", "mailing address", "mail address", "street address"), addr1),
            # street
            "2nd_line": (("",),  # TODO: find something "nice" for this line in osm data
                         lambda: s2.join([self.rand_word(), str(self._rand.randint(0, 1000))][::rc((1, -1))])),
            # postbox, building number etc...
            "city": (("",), addr3),  # city, postcode, state
            "country": (("country",), lambda: self.get_country()),
            "phone": (("tel", "phone", "phone number"), lambda: getattr(self._f, "phone_number", lambda: " ")()),
            "fax": (("fax", "fax number"), lambda: getattr(self._f, "phone_number", lambda: " ")()),
            "www": (
                ("www", "web", "homepage", "webpage", "internet"), lambda: self.generate_url(company, self._f.tld())),
            "email": (("email", ""), lambda: self._f.company_email()),
            "bank": (("bank details", "bank"),
                     lambda: " ".join(self.rand_word() for i in range(self._rand.randint(1, 5)))),
            "outro": (("©", "Copyright", "Copyright ©"),
                      lambda: rand_exec(r, lambda: self._f.year() + " ", 0.5) + company)
        }
        multi_parts = ["space"]

        part_structure: tuple[tuple[str, float], ...]
        if self._type == "contact":
            min_parts = 1
            part_structure = (
                ("intro", 0.1),
                ("name", 0.1),
                ("company", 0.2),
                ("phone", 0.3),
                ("fax", 0.1),
                ("www", 0.3),
                ("email", 0.3),
                ("bank", 0.05),
                ("outro", 0.05)
            )
            # choose only a single line in 50% of the time:
            if r() < 0.5:
                part_structure = ((rc(part_structure)[0], 1.0),)
        elif self._type == "address":
            large_address = 0.0 if r() < 0.3 else 0.3
            if r() < 0.3:  # make sure we have "short" addresses
                part_structure = (
                    ("company", 0.1),
                    ("street_address", 1.0),
                    ("city", 1.0),
                    ("country", 0.1)
                )
                min_parts = 2
            else:
                min_parts = 3
                part_structure = (
                    # (probability, fieldnames, generation function)
                    ("intro", 0.05),
                    # for example an announcement, or company introduction, catch phrase...
                    ("name", 0.1 + large_address),
                    ("space", 0.05),
                    ("company", 0.3 + large_address),
                    ("street_address", 0.95),
                    ("2nd_line", 0.05),
                    ("city", 0.95),
                    ("country", 0.2 + large_address),
                    ("space", 0.05),
                    ("phone", 0.2 + large_address),
                    ("fax", 0.1 + large_address),
                    ("www", 0.2 + large_address),
                    ("email", 0.1 + large_address),
                    ("bank", 0.05 + large_address),
                    ("outro", 0.1 + large_address)
                )
        else:
            raise NotImplementedError()

        # or a random combination
        parts = set()
        # select random selection of address parts
        while len(parts) < min_parts:  # make sure addresses consist of a min-length
            for part_id, prob in part_structure:
                if (part_id not in parts) or (part_id in multi_parts):  # make sure every part gets added only once
                    if r() < prob:
                        parts.add(part_id)

        # render the actual address
        render_parts = []
        for part_id, (field_names, v) in all_parts.items():
            if part_id in parts:
                # try:
                #    v = v()
                # except:
                #    logger.exception("error in address generation function")
                #    v = " "
                v = v()
                if field_names[0]:  # if we have anything else than an empty string..
                    # add field_name or random word as field name
                    if r() < self._fieldname_prob:
                        field_name = rc(field_names) if r() < 0.5 else self.rand_word()
                        v = field_name + rc((" ", ": ", ":")) + v

                render_parts.append(v)

        if r() < 0.1:  # randomize address line in 10%
            random.shuffle(render_parts)
        # if r() < 0.01:  # TODO: duplicate address lines, occasionally

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
        list_indicators = [""] + list(" -*∙•+~►>") + ['->']
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


class TextBlockGenerator(torch.utils.data.IterableDataset):
    def __init__(
            self,
            generators: dict[str, tuple[typing.Any]],
            random_char_prob: float = 0.0,
            random_word_prob: float = 0.0,
            random_upper_prob: float = 0.0,
            random_line_prob: float = 0.0,
            random_separation_prob: float = 0.0,
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
        self._random_char_prob = random_char_prob
        self._random_word_prob = random_word_prob
        self._random_upper_prob = random_upper_prob
        self._random_line_prob = random_line_prob
        self._random_separation_prob = random_separation_prob
        self._mixed_blocks_generation_prob = mixed_blocks_generation_prob
        self._mixed_blocks_label = mixed_blocks_label
        self._cache_size = cache_size
        self._renew_num = renew_num
        self._virtual_size = virtual_size

    @classmethod
    def std_generator(cls):
        bg = cls(
            generators={
                "address": ((100, BusinessAddressGenerator(
                    type="address", rand_str_perc=0.3, osm_perc=0.5, fieldname_prob=0.05)),),
                "unknown": ((75, RandomTextBlockGenerator(
                    txt_source=settings.TRAINING_DATA_DIR / "all_text.txt")),
                            (20, RandomListGenerator()),
                            (5, BusinessAddressGenerator(
                                type="contact", rand_str_perc=0.0, osm_perc=0.0, fieldname_prob=0.3)),)
            },
            random_char_prob=0.0025, random_word_prob=0.1, random_upper_prob=0.2, random_line_prob=0.1,
            random_separation_prob=0.2,
            cache_size=100, renew_num=10, mixed_blocks_generation_prob=0.025, mixed_blocks_label="unknown"
        )

        return bg

    @cached_property
    def rand(self):
        return random.Random(time.time())  # get our own random number generator

    @cached_property
    def available_locales(self) -> list[str]:
        import faker
        available_locales = faker.config.AVAILABLE_LOCALES
        try:
            # this in order to avoid warning: "UserWarning: fr_QC locale is deprecated. Please use fr_CA."
            available_locales.remove("fr_QC")
        except ValueError:
            pass

        return available_locales

    @cached_property
    def faker(self):
        import faker
        # pre-initialize fakers for all languages
        faker = faker.Faker(self.available_locales)
        return faker

    @cached_property
    def class_gen(self):
        return tuple(k[1] for _, j in self._generators.items() for k in j)

    @cached_property
    def gen_mapping(self):
        return {k[1]: name for name, j in self._generators.items() for k in j}

    @cached_property
    def weights(self):
        return tuple(k[0] for name, j in self._generators.items() for k in j)

    @cached_property
    def num_generators(self):
        return len(self.class_gen)

    @cached_property
    def classmap(self):
        return dict(enumerate(self._generators))

    @cached_property
    def classmap_inv(self):
        return {j: i for i, j in self.classmap.items()}

    def single(self, seed, convert_labels=False):
        # TODO: does it really make sense here to keep seed?
        #       only makes sense if EVERY random choice is 100% included in it
        self.rand.seed(seed)

        # randomly choose a generator for a certain class
        gen = self.rand.choices(population=self.class_gen, weights=self.weights)[0]
        cl: str = self.gen_mapping[gen]
        # generate random input data for the class to be predicted
        x: str = gen(seed)
        # whether we should mix up blocks occasionally
        if self._mixed_blocks_generation_prob > self.rand.random():
            gen = self.rand.choices(population=self.class_gen, weights=self.weights)[0]
            x += "\n" + gen(seed + 1)
            y = self._mixed_blocks_label
        else:
            y = cl

        if convert_labels:
            y = self.classmap_inv[y]

        # TODO: make word & character generation more aware of the "kind" of
        #       characters:  for example replace numbers only with other numbers
        #       or replace words with random, alphabet-only characters
        #       or replace with similar looking combinations
        #      (e.g. ###.###AABBC pr somthing similar)

        # word augmentation
        if self._random_word_prob:
            # we use some augmentation for our dataset here to make the classifier more robust...
            t = []
            # we would like to preserve newline chars thats why we are splitting in such a "funny" way
            words = x.replace("\n", " \n ").split(" ")
            wordnum = len(words)
            for i, w in enumerate(words):
                # TODO:  merge word and char augmentation into a general "list-augmenation" function
                if w and (self.rand.random() < self._random_word_prob):
                    selector = self.rand.randint(0, 10)
                    if selector == 0:  # substitute with random char sequences
                        t.append(self.faker.bothify(text="?" * self.rand.randint(1, 15)))
                    elif selector == 1:  # insert before
                        t.append(self.faker.bothify(text="?" * self.rand.randint(1, 15)))
                        t.append(w)
                    elif selector == 2:  # insert after
                        t.append(w)
                        t.append(self.faker.bothify(text="?" * self.rand.randint(1, 15)))
                    elif selector == 3:  # delete
                        continue
                    elif selector == 4:  # duplicate
                        t.append(w)
                        t.append(w)
                    elif selector == 5:  # swap
                        i2 = self.rand.randrange(0, wordnum)
                        t.append(words[i2])
                        words[i2] = w
                    elif selector == 6:  # neighbourswap
                        if self.rand.random() > 0.5:
                            i2 = min(i + 1, wordnum - 1)
                        else:
                            i2 = max(i - 1, 0)
                        t.append(words[i2])
                        words[i2] = w
                    elif selector == 7:  # one-side-swap (only substitue the item with a duplicate from somewhere else in the string)
                        t.append(self.rand.choice(words))
                    # already handled by inserting random characters
                    # elif selector ==8: #randomly split words
                    #    splitnum=self._random.randrange(0,len(w))
                    #    t.extend([w[:splitnum],w[splitnum:]])
                    elif selector == 8:  # reverse
                        t.append(w[::-1])
                    elif selector == 9:  # crop
                        crop = self.rand.randrange(len(w))
                        if crop > (len(w) // 2):
                            t.append(w[crop:])
                        else:
                            t.append(w[:crop])
                    elif selector == 10:  # convert to upper case
                        if self.rand.random() > 0.5:
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
                if self.rand.random() < self._random_char_prob:
                    selector = self.rand.randint(0, 9)
                    if selector == 0:  # substitute
                        t.append(self.rand.choice(_asciichars))
                    elif selector == 1:  # insert before
                        t.append(self.rand.choice(_asciichars))
                        t.append(c)
                    elif selector == 2:  # insert after
                        t.append(c)
                        t.append(self.rand.choice(_asciichars))
                    elif selector == 3:  # delete
                        continue
                    elif selector == 4:  # duplicate
                        t.append(c)
                        t.append(c)
                    elif selector == 5:  # swap
                        i2 = self.rand.randrange(0, lenx)
                        t.append(x_list[i2])
                        x_list[i2] = c
                    elif selector == 6:  # neighbourswap
                        if self.rand.random() > 0.5:
                            i2 = min(i + 1, lenx - 1)
                        else:
                            i2 = max(i - 1, 0)
                        t.append(x_list[i2])
                        x_list[i2] = c
                    elif selector == 7:  # one-side-swap (only substitue the item with a duplicate from somewhere else in the string)
                        t.append(self.rand.choice(x))
                    elif selector == 8:
                        t.append(c.upper())
                    elif selector == 9:
                        t.append(c)
                        if self.rand.random() < 0.7:
                            t.append("</s>")
                        else:
                            t.append("<s>")
                else:
                    t.append(c)

            x = "".join(t)

        # whole text augmentation
        if self.rand.random() < self._random_upper_prob:
            selec = self.rand.random()
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

        # line augmentation
        if self._random_line_prob:
            if self.rand.random() < self._random_line_prob:
                # randomly swap lines
                lines = x.split("\n")
                a, b = random.randrange(0, len(lines)), random.randrange(0, len(lines))
                lines[a], lines[b] = lines[b], lines[a]
                x = "\n".join(lines)
            # TODO: duplication
            # if self.rand.random() < self._random_line_prob:
            # randomly duplicate lines
            #    lines = x.split("\n")
            #    a, b = random.randrange(0, len(lines)), random.randrange(0, len(lines))
            #    lines[a], lines[b] = lines[b], lines[a]
            #    x = "\n".join(lines)

        if self._random_separation_prob:
            if self.rand.random() < self._random_separation_prob:
                x = x.replace("\n", self.rand.choice(("\n ", " \n", " \n ")))
            # TODO: replace separation chars with something different!

        return x, y

    def __getitem__(self, item):
        return self.single(seed=item)

    def __iter__(self):
        # initialize cache with random samples
        cache = collections.deque(
            (self.single(
                seed=self.rand.random()) for i in range(self._cache_size)),
            maxlen=self._cache_size)
        while True:
            # serve all sample currently in cache
            yield from cache
            # add new samples to the cache collections deque automatically drops old samples
            # that exceed _cache_size
            cache.extend(self.single(seed=self.rand.random()) for i in range(self._renew_num))

    def __len__(self):
        return self._virtual_size


def random_chunks(li, prob_dist):
    """splits a list into random chunks with sizes according to prob_func"""
    it = iter(li)
    while True:
        nxt = list(islice(it, prob_dist()))
        if nxt:
            yield nxt
        else:
            break


def random_chunks_uniform(li, min_size=1, max_size=20):
    """splits a list into random uniformly sized chunks"""
    prob_dist = lambda: randint(min_size, max_size)
    return random_chunks(li, prob_dist)
