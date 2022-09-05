from document import Extractor
import functools

class ExtractKeywords(Extractor):
    @property
    def keywords(self) -> list[str]:
        """a list of  keywords sometimes they are generated, other times
        they need to be extracted from the docment metadata"""
        return []

    @functools.lru_cache
    def textrank_keywords(self, k=30, max_links=4, max_distance=0.2, method="noun_chunks"):
        """extract keywords by textrank from a similarity graph of a spacy document"""
        G = self.similarity_graph(k=max_links, max_distance=max_distance, method=method)
        spans = getattr(self, method)
        keywords = ((spans[i], score) for i, score in nx.pagerank(G, weight='weight').items())
        return sorted(keywords, key=lambda x: x[1], reverse=True)[:k]