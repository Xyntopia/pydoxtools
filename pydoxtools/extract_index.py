import functools
from document import Extractor

class IndexExtractor(Extractor):
    """
    Class extracts an index form a document
    TODO: make it flexible which kind of index we can use for this :).

    --> for example we could also use a scikit-learn knn index or some brute-force method
        etc....
    """
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