import functools
from abc import ABC
from functools import cached_property
from pathlib import Path
from typing import List, Dict, Any, Union, BinaryIO, Tuple

import hnswlib
import langdetect
import networkx as nx
import numpy as np
import pandas as pd
import spacy.tokens

from pydoxtools import models, nlp_utils, classifier
from pydoxtools.list_utils import group_by


class TokenCollection:
    def __init__(self, tokens: List[spacy.tokens.Token]):
        self._tokens = tokens

    @cached_property
    def vector(self):
        return np.mean([t.vector for t in self._tokens], 0)

    @cached_property
    def text(self):
        return self.__str__()

    def __len__(self):
        return len(self._tokens)

    def __getitem__(self, item):
        return self._tokens[item]

    def __str__(self):
        return " ".join(t.text for t in self._tokens)

    def __repr__(self):
        return "|".join(t.text for t in self._tokens)


class DocumentLoader(ABC):
    """
    This class is the base for all document classes in pydoxtools and
    defines a common interface for all.

    This class also defines a basic extraction schema which derived
    classes can override
    """

    def __init__(
            self,
            fobj: Union[str, Path, BinaryIO],
            source: Union[str, Path],
            spacy_model: str = "",
            model_size: str = "",
            # Where does the extracted data come from? (Examples: URL, 'pdfupload', parent-URL, or a path)"
    ):
        """
        ner model:

        if a "spacy_model" was specified use that.
        else if "model_size" was specified, use generic spacy language model
        else  use generic, multilingual ner model "xx_ent_wiki_sm"
        """
        self._fobj = fobj
        self._source = source
        self._spacy_model = spacy_model
        self._model_size = model_size

    def __repr__(self):
        return f"{self.__module__}.{self.__class__.__name__}({self._fobj},{self.source})>"

    @property
    def type(self):
        return 'unknown'

    @cached_property
    def spacy_model_id(self) -> str:
        if self._spacy_model:
            nlpmodelid = self._spacy_model
        elif self._model_size:
            nlpmodelid = nlp_utils.get_spacy_model_id(self.lang, self._model_size)
        else:
            nlpmodelid = "xx_ent_wiki_sm"

        return nlpmodelid

    @cached_property
    def spacy_nlp(self) -> spacy.Language:
        nlp = nlp_utils.load_cached_spacy_model(self.spacy_model_id)
        return nlp

    @cached_property
    def spacy_doc(self) -> spacy.tokens.Doc:
        return self.spacy_nlp(self.full_text)

    @property
    def vectors(self):
        # TODO: also use vectors without context! (using BERT embeddings for example...)
        # TODO: this is the "old" method without spacy... we might consider
        #       to use this method again in order to leverage more huggingface models such as the multilingual one...
        #       the problem is a little bit, that "custom" huggingface mdoels don't 100% align with
        #       the spacy tokens...
        # model,tokenizer = nlp_utils.load_models()
        # v = nlp_utils.longtxt_embeddings_fullword(doc.full_text, model, tokenizer)
        # v[1]
        # pd.DataFrame(list(zip(v[1],df['text'])), columns=['trf','tok']).head(50)
        return self.spacy_doc._.trf_token_vecs

    @property
    def trf_embeddings(self):
        trfc = self.spacy_nlp.components[0][1]
        return trfc.model.transformer.embeddings.word_embeddings

    @cached_property
    def noun_chunks(self) -> List[TokenCollection]:
        token_list = []
        for nc in self.spacy_doc.noun_chunks:
            tc = TokenCollection([t for t in nc if t.pos_ not in ["DET", "SPACE", "PRON"]])
            if len(tc) > 0:
                token_list.append(tc)
        # filter = ["DET"]
        return token_list

    @cached_property
    def sents(self) -> List[spacy.tokens.Span]:
        return list(self.spacy_doc.sents)

    @functools.lru_cache()
    def index(self, filter: str = ""):
        """create a nearest neighbour search index

        pos_filter creates an index only for

        TODO: document this function a little better
        """
        index = hnswlib.Index(space='cosine', dim=self.vectors.shape[1])
        if filter == "noun_chunks":
            vecs = [e.vector for e in self.noun_chunks]
            idx = list(range(len(vecs)))

            index.init_index(max_elements=len(vecs) + 1, ef_construction=200, M=16)
            # Element insertion (can be called several times):
            index.add_items(data=vecs, ids=idx)
        elif filter == "sents":
            vecs = [s.vector for s in self.sents]
            idx = list(range(len(vecs)))

            index.init_index(max_elements=len(vecs) + 1, ef_construction=200, M=16)
            # Element insertion (can be called several times):
            index.add_items(data=vecs, ids=idx)
        elif filter:
            token_list = [t for t in self.spacy_doc if (t.pos_ in filter)]
            vecs = [t.vector for t in token_list]
            idx = [t.i for t in token_list]

            index.init_index(max_elements=len(vecs) + 1, ef_construction=200, M=16)
            # Element insertion (can be called several times):
            index.add_items(data=vecs, ids=idx)
        else:
            # max_elements defines the maximum number of elements that can be stored in
            # the structure(can be increased/shrunk).
            # ef_construction: quality vs speed parameter
            # M = max number of outgoing connections in the graph...
            index.init_index(max_elements=len(self.spacy_doc) + 1, ef_construction=200, M=16)
            # Element insertion (can be called several times):
            index.add_items(data=self.vectors, ids=list(t.i for t in self.spacy_doc))
            # Controlling the recall by setting ef:
            # index.set_ef(100)  # ef should always be > k

        return index

    def knn_query(
            self,
            txt: Union[str, spacy.tokens.Token, np.ndarray, TokenCollection],
            k: int = 5, filter="", indices=False
    ) -> List[Tuple]:
        if isinstance(txt, str):
            search_vec = self.spacy_nlp(txt).vector
        elif isinstance(txt, np.ndarray):
            search_vec = txt
        else:
            search_vec = txt.vector
        similar = self.index(filter=filter).knn_query([search_vec], k=k)
        if filter == "noun_chunks":
            wordlist = self.noun_chunks
        elif filter == "sents":
            wordlist = self.sents
        else:  # default behaviour
            wordlist = self.spacy_doc

        if indices:
            return [(i, wordlist[i], dist) for i, dist in zip(similar[0][0], similar[1][0])]
        else:
            return [(wordlist[i], dist) for i, dist in zip(similar[0][0], similar[1][0])]

    # @cached_property
    @functools.lru_cache
    def similarity_graph(self, k=4, max_distance=0.2, method="noun_chunks"):
        """
        this function buils a "directed similarity graph" by taking the similarity of words in a document
        and connecting tokens which are similar. This can then be used for further analysis
        such as textrank (wordranks, sentence ranks, paragraph ranking) etc...
        """
        G = nx.DiGraph()
        if method == "noun_chunks":
            source = self.noun_chunks
        elif method == "sents":
            source = self.sents
        else:
            raise NotImplementedError(f"can not have method: '{method}'")

        for i, span in enumerate(source):
            G.add_node(i, label=span.text)
            # we take k+1 here, as the first element will always be the query token itself...
            similar = self.knn_query(span, k=k + 1, filter=method, indices=True)
            # links = links[links<0.3]
            doc_sim = nlp_utils.cos_similarity(self.spacy_doc.vector[None, :], span.vector[None, :])[0]
            for j, _, d in similar:
                if (not i == j) and (d <= max_distance):
                    G.add_edge(i, j, weight=(1 - d))
        return G

    @functools.lru_cache
    def textrank_keywords(self, k=30, max_links=4, max_distance=0.2, method="noun_chunks"):
        """extract keywords by textrank from a similarity graph of a spacy document"""
        G = self.similarity_graph(k=max_links, max_distance=max_distance, method=method)
        spans = getattr(self, method)
        keywords = ((spans[i], score) for i, score in nx.pagerank(G, weight='weight').items())
        return sorted(keywords, key=lambda x: x[1], reverse=True)[:k]

    # TODO: save document structure as a graph...
    # nx.write_graphml_lxml(G,'test.graphml')
    # nx.write_graphml(G,'test.graphml')

    @property
    def model(self) -> models.DocumentExtract:
        data = models.DocumentExtract.from_orm(self)
        return data

    # TODO: more configuration options:
    #       - which nlp models (spacy/transformers) to use
    #       - should "full text" include tables?
    #       - should ner include tables/figures?

    # TODO: calculate md5-hash for the document and
    #       use __eq__ with that hash...
    #       we need this for caching purposes but also in order
    #       check if a document already exists...

    # TODO: test this for path, string, fobj and string path for different
    #       documents
    @property
    def filename(self) -> str:
        if isinstance(self._fobj, str):
            return str(self._fobj)
        elif isinstance(self._fobj, Path):
            return self._fobj.name
        else:
            return self._fobj.name

    @property
    def source(self) -> str:
        return self._source

    @property
    def fobj(self) -> Union[str, BinaryIO]:
        return self._fobj

    @property
    def mime_type(self) -> str:
        """
        type such as "pdf", "html" etc...  can also be the mimetype!
        TODO: maybe we can do something generic here?
        """
        return "unknown"

    @property
    def list_lines(self):
        return []

    @property
    def tables(self) -> List[Dict[str, Dict[str, Any]]]:
        """
        table in the following (row - wise) format:

        [{index -> {column -> value } }]
        """
        return []

    @property
    def tables_df(self) -> List["pd.DataFrame"]:
        return []

    @cached_property
    def lang(self) -> str:
        text = self.full_text.strip()
        if text:
            lang = langdetect.detect(text)
        else:
            lang = "unknown"
        return lang

    @property
    def textboxes(self) -> List[str]:
        return []

    @cached_property
    def full_text(self) -> str:
        return ""

    @cached_property
    def pages(self) -> list[str]:
        """automatically divide text into approx. pages"""
        page_word_size = 500
        words = self.full_text.split()
        # for i in range(len(words)):
        pages = list(words[i:i + page_word_size] for i in range(0, len(words), page_word_size))
        return pages

    @cached_property
    def num_pages(self) -> int:
        return len(self.pages)

    @cached_property
    def ner(self) -> List[Tuple[str, str]]:
        # TODO: add transformers as ner recognition as well:
        #       from transformers import pipeline
        #
        # #ner_pipe = pipeline("ner")
        # #good results = "xlm-roberta-large-finetuned-conll03-english" # large but good
        # #name = "sshleifer/tiny-dbmdz-bert-large-cased-finetuned-conll03-english" #small and bad
        # name = "Davlan/distilbert-base-multilingual-cased-ner-hrl"
        # model = name
        # tokenizer= name
        # ner_pipe = pipeline(task="ner", model=model, tokenizer=tokenizer)
        # TODO: replace the below by self.model
        if self._model_size:
            model_id = nlp_utils.get_spacy_model_id(self.lang, self._model_size)
            nlp = nlp_utils.load_cached_spacy_model(model_id)
        else:
            nlp = nlp_utils.load_cached_spacy_model(self._spacy_model)

        res = nlp_utils.extract_entities_spacy(self.full_text, nlp)
        return res

    @cached_property
    def grouped_ner(self) -> Dict[str, Any]:
        """Group labels from named entity recognition."""
        groups = group_by([n[::-1] for n in self.ner])
        return groups

    @cached_property
    def urls(self) -> List[str]:
        urls = nlp_utils.get_urls_from_text(self.full_text)
        return urls

    @cached_property
    def text_block_classes(self) -> pd.DataFrame:
        model: classifier.txt_block_classifier = classifier.load_classifier('text_block')
        blocks = self.textboxes
        classes = model.predict_proba(blocks).numpy()
        txtblocks = pd.DataFrame(self.textboxes, columns=["txt"])
        txtblocks[["add_prob", "ukn_prob"]] = classes.round(3)
        return txtblocks

    @cached_property
    def addresses(self) -> List[str]:
        return self.text_block_classes[self.text_block_classes['add_prob'] > 0.5].txt.tolist()

    @property
    def images(self) -> List:
        return []

    @property
    def titles(self) -> List[str]:
        return []

    @property
    def docinfo(self) -> List[Dict[str, str]]:
        """list of document metadata such as author, creation date, organization"""
        return []

    @property
    def meta_infos(self) -> Dict:
        # specify metainfos in a better way
        return {}

    @property
    def raw_content(self) -> List[str]:
        """for example the raw html string in the case of an html document or the raw text for markdown"""
        return []

    @property
    def keywords(self) -> List[str]:
        """a list of  keywords sometimes they are generated, other times
        they need to be extracted from the docment metadata"""
        return []

    @property
    def final_url(self) -> List[str]:
        """sometimes, a document points to a url itself (for example a product webpage) and provides
        a link where this document can be found. And this url does not necessarily have to be the same as the source
        of the document."""
        return []

    @property
    def schemadata(self) -> Dict:
        """schema.org data extracted from html meta tags and other metainfos from documents

        TODO: more detailed description of return type"""
        return {}

    @property
    def product_ids(self) -> Dict[str, str]:
        return {}

    @property
    def pdf_links(self) -> List[str]:
        """sources that embed this document as a link (for example a product page which embeds
        a link to this document (e.g. a datasheet)

        TODO: rename to "parent_source" """
        return []

    @property
    def price(self) -> List[str]:
        """if prices are given in the document"""
        return []
