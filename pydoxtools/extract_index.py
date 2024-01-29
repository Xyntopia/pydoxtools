from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import logging
import uuid
from typing import Callable

import dask.bag
import hnswlib
import networkx as nx
import numpy as np
import pydantic

import pydoxtools
from pydoxtools.document_base import TokenCollection
from pydoxtools.nlp_utils import calculate_string_embeddings
from pydoxtools.operators_base import Operator
from . import list_utils

logger = logging.getLogger(__name__)


class TextPieceSplitter(Operator):
    """Extract small pieces of text with a few more rules
    to make them more usable in an index.

    For example, it is a good idea to make text-pieces not too small. Text pieces should
    also not bee too big. So that they encode not too much information
    into the vector. This makes queries on an index more precise.
    Additionally, we should try to segment a piece of text into its
    logical structure and try to preserve text blocks such as paragraphs,
    tables etc... as much as possible.

    if we split up large text blocks we will let the individual pieces overlap
    just a little bit in order to preserve some of the context.
    """

    def __call__(
            self, full_text, min_size: int = 256, max_size: int = 512,
            large_segment_overlap=0.3, max_text_segment_num=100,
    ) -> list[str]:
        # TODO: also accept text elements which have bounding boxes for better text box identification.
        # TODO: identify tables and convert them into a readable format.
        # TODO: identify other things such as images, plots etc..  and convert them into
        #       a vector
        # TODO: use this approach for faster questions-answering!
        split_text = full_text.split("\n\n")

        # merge text_pieces that are too small:
        pieces = []
        new_segment = ""
        for tp in split_text:
            if len(new_segment) < min_size:
                new_segment += tp + "\n\n"
                continue

            new_segment += tp
            if len(new_segment) > max_size:
                new_segments = []
                for i in range(0, len(new_segment), int((1 - large_segment_overlap) * max_size)):
                    new_segments.append(new_segment[i:i + max_size].strip())
                pieces.extend(new_segments)
            else:
                pieces.append(new_segment.strip())
            new_segment = ""

            if len(pieces) > max_text_segment_num:
                break

        # Append the last piece if it's not empty
        if new_segment.strip():
            pieces.append(new_segment.strip())

        return pieces


class HuggingfaceVectorizer(Operator):
    """Vectorize text pieces using tokenizers and models from huggingface"""

    def __call__(self, text_segments: list[str], model_id: str, only_tokenizer: bool):
        # TODO: use contextual embeddings as well!
        # model = AutoModelForQuestionAnswering.from_pretrained(model_id)
        vecs = [calculate_string_embeddings(txt, model_id=model_id, only_tokenizer=True)
                for txt in text_segments]
        return vecs


class IndexExtractor(Operator):
    """
    Class extracts an index form a document
    TODO: make it flexible which kind of index we can use for this :).

    --> for example we could also use a scikit-learn knn index or some brute-force method
        etc....
    """

    def __call__(self, vecs: np.ndarray, ids: list[int]):
        """create a nearest neighbour search index

        pos_filter creates an index only for specific word types

        TODO: document this function a little better
        """
        index = hnswlib.Index(space='cosine', dim=vecs.shape[1])
        # ids = list(range(len(vecs)))
        """
        if filter == "noun_chunks":
            vecs = [e.vector for e in self.noun_chunks]
            idx = list(range(len(vecs)))
        elif filter == "sents":
            vecs = [s.vector for s in self.sents]
            idx = list(range(len(vecs)))
        elif filter:
            token_list = [t for t in self.spacy_doc if (t.pos_ in filter)]
            vecs = [t.vector for t in token_list]
            idx = [t.i for t in token_list]
        else:
            vecs = self.vecs
            idx = list(t.i for t in self.spacy_doc)
        """

        # max_elements defines the maximum number of elements that can be stored in
        # the structure(can be increased/shrunk).
        # ef_construction: quality vs speed parameter
        # M = max number of outgoing connections in the graph...
        index.init_index(max_elements=len(vecs) + 1, ef_construction=200, M=16)
        # Element insertion (can be called several times):
        index.add_items(data=vecs, ids=ids)
        # Controlling the recall by setting ef:
        # index.set_ef(100)  # ef should always be > k

        return index


class KnnQuery(Operator):
    def __call__(
            self,
            index: hnswlib.Index,
            idx_values: list,
            vectorizer: Callable,  # e.g. lambda spacy_nlp, txt: spacy_nlp spacy_nlp(txt).vector
    ) -> Callable[..., list[tuple]]:
        def knn_query(txt: str | np.ndarray | TokenCollection, k: int = 5, indices=False) -> list[tuple]:
            if isinstance(txt, str):
                search_vec = vectorizer(txt)
            elif isinstance(txt, np.ndarray):
                search_vec = txt
            else:
                search_vec = txt.vector

            k = min(len(idx_values),k)
            try:
                similar = index.knn_query([search_vec], k=k)
            except RuntimeError:  # text is probably too small
                return []

            if indices:
                return [(i, idx_values[i], dist) for i, dist in zip(similar[0][0], similar[1][0])]
            else:
                return [(idx_values[i], dist) for i, dist in zip(similar[0][0], similar[1][0])]

        return knn_query


class SimilarityGraph(Operator):
    """
     this function buils a "directed similarity graph" by taking the similarity of words in a document
     and connecting tokens which are similar. This can then be used for further analysis
     such as textrank (wordranks, sentence ranks, paragraph ranking) etc...
     """

    def __init__(self, max_connectivity=4, max_distance=0.2):
        super().__init__()
        self.k = max_connectivity
        self.max_distance = max_distance

    def __call__(self, index_query_func: Callable, source: list[TokenCollection]):
        G = nx.DiGraph()
        for i, token_span in enumerate(source):
            G.add_node(i, label=token_span.text)
            # we take k+1 here, as the first element will always be the query token itself...
            similar = index_query_func(token_span, k=self.k + 1, indices=True)
            # links = links[links<0.3]
            # doc_sim = nlp_utils.cos_similarity(self.spacy_doc.vector[None, :], span.vector[None, :])[0]
            for j, _, d in similar:
                if (not i == j) and (d <= self.max_distance):
                    G.add_edge(i, j, weight=(1 - d))
        return G


class TextrankOperator(Operator):
    def __call__(self, G: nx.Graph, top_k: int) -> set[str]:
        """extract keywords by textrank from a similarity graph of a spacy document"""
        nodes = ((G.nodes[i]["label"], score) for i, score in nx.pagerank(G, weight='weight').items())
        top_nodes = sorted(nodes, key=lambda x: x[1], reverse=True)[:top_k]
        seen = set()  # use set to remove duplicates
        seen_add = seen.add
        return [x[0] for x in top_nodes if not (x[0] in seen or seen_add(x[0]))]