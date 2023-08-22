import functools
import logging
import subprocess
import typing
from typing import Optional, Any

import numpy as np
import pandas as pd
import spacy
import html
import torch
from spacy import Language
from spacy.tokens import Doc, Token, Span

import networkx as nx

from . import list_utils
from .document_base import TokenCollection
from .operators_base import Operator

logger = logging.getLogger(__name__)


def download_model(model_id: str):
    # python -m spacy download en_core_web_sm
    return subprocess.call(['python', '-m', 'spacy', 'download', model_id])
    # spacy.load("en_core_web_sm")


def extract_noun_chunks(spacy_doc) -> typing.List[TokenCollection]:
    token_list = []
    for nc in spacy_doc.noun_chunks:
        tc = TokenCollection([t for t in nc if t.pos_ not in ["DET", "SPACE", "PRON"]])
        if len(tc) > 0:
            token_list.append(tc)
    # filter = ["DET"]
    return token_list


def extract_spacy_token_vecs(spacy_doc) -> torch.Tensor | Any:
    if spacy_doc.has_vector:
        return spacy_doc.tensor
    else:
        return spacy_doc._.trf_token_vecs


def get_spacy_embeddings(spacy_nlp):
    try:
        return spacy_nlp.components[0][1].model.transformer.embeddings.word_embeddings
    except AttributeError:
        # not sure how to get embeddings from here:      t2v = spacy_nlp.components[0][1]
        return []


def get_spacy_model_id(model_language, size="sm") -> Optional[str]:
    """
    size can be: sm, md, lg or trf where "trf" is transformer

    TODO: try to list all spacy models systematically
     """
    if model_language in ['en', 'zh']:
        return f'{model_language}_core_web_{size}'
    elif model_language in ['de', 'fr', 'es', 'ru', 'ja', 'it', 'ca', 'hr', 'da',
                            'nl', 'fi', 'el', 'ko', 'lt', 'mk', 'nb', 'pl', 'pt',
                            'ro', 'ru', 'sv', 'uk']:
        return f'{model_language}_core_news_{size}'
    else:
        return f'xx_sent_ud_sm'


@functools.lru_cache
def load_cached_spacy_model(model_id: str) -> Language:
    """
    load spacy nlp model and in case of a transformer model add custom vector pipeline...

    we also make sure to cache the model for batch operations on documens.
    """
    try:
        logger.info(f"loading spacy model: {model_id}")
        nlp = spacy.load(model_id)
    except OSError:  # model doesn't seem to be present, yet
        logger.info(f"failed, loading. trying to download spacy model: {model_id}")
        download_model(model_id)
        nlp = spacy.load(model_id)

    if model_id[-3:] == "trf":
        nlp.add_pipe('trf_vectors')

    return nlp


@Language.factory('trf_vectors')
class TrfContextualVectors:
    """
    Spacy pipeline which add transformer vectors to each token based on user hooks.

    https://spacy.io/usage/processing-pipelines#custom-components-user-hooks
    https://github.com/explosion/spaCy/discussions/6511
    """

    def __init__(self, nlp: Language, name: str):
        # TODO: we can configure this class for different pooling methods...
        self.name = name
        Doc.set_extension("trf_token_vecs", default=None)

    def __call__(self, sdoc):
        # inject hooks from this class into the pipeline
        if type(sdoc) == str:
            sdoc = self._nlp(sdoc)

        # pre-calculate all vectors for every token:

        # calculate groups for spacy token boundaries in the trf vectors
        vec_idx_splits = np.cumsum(sdoc._.trf_data.align.lengths)
        # get transformer vectors and reshape them into one large continous tensor
        trf_vecs = sdoc._.trf_data.tensors[0].reshape(-1, 768)
        # calculate mapping groups from spacy tokens to transformer vector indices
        vec_idxs = np.split(sdoc._.trf_data.align.dataXd, vec_idx_splits)

        # take sum of mapped transformer vector indices for spacy vectors
        # TOOD: add more pooling methods than just sum...
        #       if we do this we probabyl need to declare a factory function...
        vecs = np.stack([trf_vecs[idx].sum(0) for idx in vec_idxs[:-1]])
        sdoc._.trf_token_vecs = vecs

        sdoc.user_token_hooks["vector"] = self.token_vector
        sdoc.user_span_hooks["vector"] = self.span_vector
        sdoc.user_hooks["vector"] = self.doc_vector
        sdoc.user_token_hooks["has_vector"] = self.has_vector
        sdoc.user_span_hooks["has_vector"] = self.has_vector
        sdoc.user_hooks["has_vector"] = self.has_vector
        # sdoc.user_token_hooks["similarity"] = self.similarity
        # sdoc.user_span_hooks["similarity"] = self.similarity
        # sdoc.user_hooks["similarity"] = self.similarity
        return sdoc

    @functools.lru_cache
    def token_vector(self, token: Token):
        return token.doc._.trf_token_vecs[token.i]

    @functools.lru_cache
    def span_vector(self, span: Span):
        vecs = span.doc._.trf_token_vecs
        return vecs[span.start: span.end].sum(0)

    @functools.lru_cache
    def doc_vector(self, doc: Doc):
        vecs = doc._.trf_token_vecs
        return vecs.sum(0)

    def has_vector(self, token):
        return True


class SpacyOperator(Operator):
    def __call__(
            self,
            full_text: str,
            language: str,
            spacy_model: str,
            model_size: str
    ) -> typing.Dict[str, Language | Doc]:
        """Load a document using spacy"""
        if spacy_model == "auto":
            nlp_modelid = get_spacy_model_id(language, model_size)
        else:
            nlp_modelid = spacy_model

        spacy_nlp = load_cached_spacy_model(nlp_modelid)
        return dict(
            doc=spacy_nlp(full_text),
            nlp=spacy_nlp
        )


def graphviz_sanitize(text: str) -> str:
    """sanitize text for use in graphviz"""
    text = html.escape(text).replace('\n', '<br/>')
    return text


def graphviz_prepare(token: spacy.tokens.Token, ct_size=100) -> str:
    """sanitize text for use in graphviz"""
    text = token.text
    text = graphviz_sanitize(text)
    document_text = token.doc.text
    n = token.idx
    if ct_size:
        ct = document_text[max(0, n - ct_size):n + ct_size]
        # ct = graphviz_sanitize(ct)
        ct = (f'<<font point-size="8">{graphviz_sanitize(ct[:ct_size])}</font>'
              f'<font point-size="15">{graphviz_sanitize(ct[ct_size:len(text) + ct_size])}</font>'
              f'<font point-size="8">{graphviz_sanitize(ct[len(text) + ct_size:])}</font>>')
        tx = ct
    else:
        tx = f'<<font point-size="15">{text}</font>>'
    return tx


class ExtractRelationships(Operator):
    def __call__(
            self, spacy_doc: Doc
    ) -> pd.DataFrame:
        """Extract some relationships of a spacy document for use in a knowledge graph"""
        relationships = []

        tok: spacy.tokens.Token
        for tok in spacy_doc:
            if tok.dep_ in ('nsubj', 'nsubjpass'):
                for possible_object in tok.head.children:
                    if possible_object.dep_ in ('dobj', 'pobj'):  # direct object or object of preposition
                        relationships.append({
                            'n1': tok,
                            'n2': possible_object,
                            'type': 'SVO',
                            'label': tok.head.text,
                        })

            # Attribute relationships
            if tok.dep_ in {"attr"}:
                subject = [w for w in tok.head.lefts if w.dep_ == "nsubj"]
                if subject:
                    subject = subject[0]
                    relationships.append({
                        'n1': subject,
                        'n2': tok,
                        'type': 'Attribute',
                        'label': 'is',
                    })

            # Prepositional relationships
            if tok.dep_ == "pobj" and tok.head.dep_ == "prep":
                relationships.append({
                    'n1': tok.head.head,
                    'n2': tok,
                    'type': 'Prepositional',
                    'label': 'related',
                })

            # Possessive relationships
            if tok.dep_ == "poss":
                relationships.append({
                    'n1': tok,
                    'n2': tok.head,
                    'type': 'Possessive',
                    'label': 'owns',
                })

            # Adjective relationships
            if tok.dep_ == "amod":
                relationships.append({
                    'n1': tok.head,
                    'n2': tok,
                    'type': 'Adjective',
                    'label': 'is',
                })

        res = pd.DataFrame(relationships)
        if res.empty:
            res = pd.DataFrame(columns=['n1', 'n2', 'type', 'label'])
        return res


@functools.lru_cache
def load_cached_fast_coref_model(method='fast'):
    if method == 'fast':
        from fastcoref import FCoref
        model = FCoref()
    else:
        from fastcoref import LingMessCoref
        model = LingMessCoref()

    return model


class CoreferenceResolution(Operator):
    def __call__(
            self, spacy_doc: Doc,
            method='fast'
    ) -> list[list[tuple[int, int]]]:
        """Resolve coreferences in a spacy document"""
        model = load_cached_fast_coref_model(method)

        preds = model.predict(
            texts=[[str(t) for t in spacy_doc]],
            is_split_into_words=True
        )

        tok_id_coreferences = preds[0].get_clusters(as_strings=False)

        return tok_id_coreferences


def build_relationships_graph(
        relationships: pd.DataFrame,
        coreferences: list[list[tuple[int, int]]],
) -> dict[str, pd.DataFrame | pd.Series]:
    # get all nodes from the relationship list
    graph_nodes = pd.DataFrame(set(relationships[["n1", "n2"]].values.flatten()), columns=["nodes"])

    if graph_nodes.empty:
        return dict(
            graph_nodes=pd.DataFrame(columns=['nodes', 'token_idx', 'group']),
            node_map=pd.Series(),
            graph_edges=pd.DataFrame(columns=['n1', 'n2', 'type', 'label']),
        )

    # add node groups
    graph_nodes['token_idx'] = graph_nodes.nodes.apply(lambda x: x.i).astype(int)

    #
    graph_nodes.set_index('token_idx', inplace=True)

    # G[doc.spacy_doc[4]]
    coreferences = coreferences

    # add groups to nodes
    graph_nodes['group'] = graph_nodes.index  # set standard group to token index
    for i, cr_group in enumerate(coreferences):
        indices = list(list_utils.flatten(
            [list(range(idx[0], idx[1])) for idx in cr_group])
        )
        # rel.loc[indices, 'group'] = i
        group_idx = min(indices)
        valid_indices = graph_nodes.index[graph_nodes.index.isin(indices)]
        graph_nodes.loc[valid_indices, 'group'] = group_idx
        # nG = nx.identified_nodes(G, node,node2)

    # grouping all tokens
    graph_nodes = graph_nodes.groupby('group').agg({'nodes': list})

    def identify_mean_node(token_list: pd.Series):
        toks = pd.DataFrame(token_list.nodes, columns=["tok"])
        toks["text"] = toks.apply(lambda x: x[0].text, axis=1)
        most_occuring_text = toks["text"].value_counts().index[0]
        idx = toks.loc[toks.text == most_occuring_text].tok.apply(lambda x: x.i).min()
        return pd.Series(dict(
            label=most_occuring_text,
            idx=idx,
            toks=token_list.nodes
        ))

    graph_nodes = graph_nodes.apply(identify_mean_node, axis=1)

    graph_nodes['tok_idx'] = graph_nodes.toks.apply(lambda x: set([t.i for t in x]))
    node_map = graph_nodes[['idx', 'tok_idx']].explode('tok_idx').set_index('tok_idx').idx

    edges = relationships.apply(lambda r: pd.Series(dict(
        n1=node_map[r.n1.i],
        n2=node_map[r.n2.i],
        label=r.label,
        type=r.type
    )), axis=1)

    return dict(
        graph_nodes=graph_nodes,
        node_map=node_map,
        graph_edges=edges
    )


def nx_graph(
        graph_nodes: pd.DataFrame,
        graph_edges: pd.DataFrame,
        spacy_doc: Doc,
        graph_debug_context_size=0,
) -> nx.DiGraph:
    """build a graph from the relationships extracted from a spacy document"""
    KG = nx.DiGraph()

    for _, node in graph_nodes.iterrows():
        # TODO: add more debug info in case of nodes that have been merged by coreference
        # we can do this here, because we always chose a "valid" spacy-token as the representative index
        # in case of a merge, the index of the most frequent token is chosen
        t = spacy_doc[node.idx]
        tx = graphviz_prepare(t, ct_size=graph_debug_context_size)
        KG.add_node(node.idx, label=tx, shape="box")

    for _, e in graph_edges.iterrows():
        tx = graphviz_sanitize(e.label)
        KG.add_edge(e.n1, e.n2, label=tx, type=e["type"])

    return KG
