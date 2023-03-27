from typing import Callable

import hnswlib
import networkx as nx
import numpy as np

from pydoxtools.document_base import Extractor, TokenCollection


class IndexExtractor(Extractor):
    """
    Class extracts an index form a document
    TODO: make it flexible which kind of index we can use for this :).

    --> for example we could also use a scikit-learn knn index or some brute-force method
        etc....
    """

    def __init__(self):
        super().__init__()

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


class KnnQuery(Extractor):
    def __init__(self):
        super().__init__()

    def __call__(
            self,
            index: hnswlib.Index,
            idx_values: list,
            vectorizer: Callable,  # e.g. lambda spacy_nlp, txt: spacy_nlp spacy_nlp(txt).vector
    ) -> Callable:
        def knn_query(txt: str | np.ndarray | TokenCollection, k: int = 5, indices=False) -> list[tuple]:
            if isinstance(txt, str):
                search_vec = vectorizer(txt)
            elif isinstance(txt, np.ndarray):
                search_vec = txt
            else:
                search_vec = txt.vector
            similar = index.knn_query([search_vec], k=k)

            if indices:
                return [(i, idx_values[i], dist) for i, dist in zip(similar[0][0], similar[1][0])]
            else:
                return [(idx_values[i], dist) for i, dist in zip(similar[0][0], similar[1][0])]

        return knn_query


class SimilarityGraph(Extractor):
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


class ExtractKeywords(Extractor):
    def __init__(self, top_k: int):
        super().__init__()
        self.k = top_k

    def __call__(self, G: nx.Graph):
        """extract keywords by textrank from a similarity graph of a spacy document"""
        keywords = ((G.nodes[i]["label"], score) for i, score in nx.pagerank(G, weight='weight').items())
        top_kw = sorted(keywords, key=lambda x: x[1], reverse=True)[:self.k]
        return [kw[0] for kw in top_kw]
