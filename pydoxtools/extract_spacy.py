import functools
import itertools
import logging
import subprocess
import typing
from typing import Optional, Any

import networkx
import numpy as np
import pandas as pd
import pydoxtools.document_base
import spacy
import torch
from spacy import Language
from spacy.tokens import Doc, Token, Span

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

        # For spaCy v3.7+, trf pipelines use spacy-curated-transformers and doc._.trf_data
        # is a DocTransformerOutput object. --> https://spacy.io/api/curatedtransformer#doctransformeroutput
        # calculate groups for spacy token boundaries in the trf vectors
        vec_idx_splits = np.cumsum(sdoc._.trf_data.last_hidden_layer_state.lengths)
        # get transformer vectors and reshape them into one large continous tensor
        trf_vecs = sdoc._.trf_data.last_hidden_layer_state.data
        # calculate mapping groups from spacy tokens to transformer vector indices
        vec_idxs = np.split(range(0, sdoc._.trf_data.last_hidden_layer_state.data.shape[0]), vec_idx_splits)

        # take sum of mapped transformer vector indices for spacy vectors
        # TOOD: add more pooling methods than just sum...
        #       if we do this we probabyl need to declare a factory function...
        # i
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


def get_token_context(token: spacy.tokens.Token, ct_size=100) -> tuple[str, str, str]:
    if ct_size:
        document_text: str = token.doc.text
        n = token.idx
        text = token.text
        ct = document_text[:n + ct_size]
        pre_c_start = n - min(ct_size, n)
        tok_start = n
        tok_end = n + len(text)
        post_c_end = tok_end + min(ct_size, len(document_text) - n)
        # ct = graphviz_sanitize(ct)
        ct = (document_text[pre_c_start:tok_start],
              document_text[tok_start:tok_end],
              document_text[tok_end:post_c_end])
    else:
        ct = None
    return ct


class ExtractRelationships(Operator):
    def __call__(
            self, spacy_doc: Doc
    ) -> pd.DataFrame:
        """Extract some semantic_relations of a spacy document for use in a knowledge graph"""
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

            # Attribute semantic_relations
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

            # Prepositional semantic_relations
            if tok.dep_ == "pobj" and tok.head.dep_ == "prep":
                relationships.append({
                    'n1': tok.head.head,
                    'n2': tok,
                    'type': 'Prepositional',
                    'label': 'related',
                })

            # Possessive semantic_relations
            if tok.dep_ == "poss":
                relationships.append({
                    'n1': tok,
                    'n2': tok.head,
                    'type': 'Possessive',
                    'label': 'has',
                })

            # Adjective semantic_relations
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


def build_document_graph(
        semantic_relations: pd.DataFrame,
        coreferences: list[list[tuple[int, int]]],
        document_objects: dict[int, pydoxtools.document_base.DocumentElement],
        page_set: set[int],
        cts: int,  # graph context size for debugging
        meta: dict  # document metadata
) -> networkx.DiGraph:
    # generate node ids
    ids = itertools.count()

    DG = networkx.DiGraph()
    # add additional nodes & edges
    doc_id = next(ids)
    meta['label'] = "document"
    DG.add_node(doc_id, **meta)
    page_map = {}
    for p in page_set:
        p_id = next(ids)
        DG.add_node(p_id, label=f"page[{p}]", page_num=0)
        DG.add_edge(doc_id, p_id, label="is parent", type="document_hierarchy")
        page_map[p] = p_id

    # TODO: with each one of the following types
    #       we might need to be careful, how o make them "useful" for
    #       a KG. Sometimes, there might just be "too many" of one element
    allowed_types = (
        pydoxtools.document_base.ElementType.Table,
        pydoxtools.document_base.ElementType.TextBox,
        pydoxtools.document_base.ElementType.Image,
        pydoxtools.document_base.ElementType.List,
        pydoxtools.document_base.ElementType.Figure,
    )
    label_nodes_map: dict[str, int] = {}
    for id, do in document_objects.items():
        if do.type in allowed_types:
            if do.labels and (do.labels != ["unknown"]):
                do_id = next(ids)
                p = do.p_num
                node_label = do.place_holder_text
                DG.add_node(do_id, label=node_label, obj=do)
                DG.add_edge(page_map[p], do_id, label="is parent", type="document_hierarchy")
                # and add the labels as well
                for l in do.labels:
                    if l in label_nodes_map:
                        label_id = label_nodes_map[l]
                    else:
                        label_id = next(ids)
                        DG.add_node(label_id, label=l)
                        label_nodes_map[l] = label_id
                    DG.add_edge(do_id, label_id, label="is")

    if not semantic_relations.empty:
        # get all nodes from the relationship list
        graph_nodes = pd.DataFrame(set(semantic_relations[["n1", "n2"]].values.flatten()), columns=["nodes"])

        # add node groups
        graph_nodes['token_idx'] = graph_nodes.nodes.apply(lambda x: x.i).astype(int)
        graph_nodes.set_index('token_idx', inplace=True)

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
            first_token = token_list.nodes[0]
            node_attrs = dict(
                label=most_occuring_text,
                idx=next(ids),
                toks=token_list.nodes
            )
            if cts:
                node_attrs["context"] = get_token_context(first_token, ct_size=cts)
            return pd.Series(node_attrs)

        graph_nodes = graph_nodes.apply(identify_mean_node, axis=1)

        graph_nodes['tok_idx'] = graph_nodes.toks.apply(lambda x: set([t.i for t in x]))
        # node_map maps the index of a token to a graph node index
        node_map = graph_nodes[['idx', 'tok_idx']].explode('tok_idx').set_index('tok_idx').idx

        edges = semantic_relations.apply(lambda r: pd.Series(dict(
            n1=node_map[r.n1.i],
            n2=node_map[r.n2.i],
            data=dict(label=r.label,
                      type=r.type)
        )), axis=1)

        # convert graph_nodes into a dictionary
        graph_nodes = graph_nodes.set_index('idx').to_dict('index')
        DG.add_nodes_from(graph_nodes.items())
        DG.add_edges_from(edges.values)

    return DG
