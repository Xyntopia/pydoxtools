#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import logging
from enum import Enum

import networkx as nx
import spacy.tokens
from IPython.core.display import SVG

logger = logging.getLogger(__name__)


def create_example_graph():
    import pyyed
    g = pyyed.Graph()

    g.add_node('foo', font_family="Zapfino")
    g.add_node('foo2', shape="roundrectangle", font_style="bolditalic", underlined_text="true")

    g.add_edge('foo1', 'foo2')
    g.add_node('abc', font_size="72", height="100", shape_fill="#FFFFFF")

    g.add_node('bar', label="Multi\nline\ntext")
    g.add_node('foobar', label="""Multi
        Line
        Text!""")

    g.add_edge('foo', 'foo1', label="EDGE!", width="3.0", color="#0000FF",
               arrowhead="white_diamond", arrowfoot="standard", line_type="dotted")

    graphstring = g.get_graph()
    return graphstring


def save_to_file(g):
    # To write to file:
    with open('test_graph.graphml', 'w') as fp:
        fp.write(g.get_graph())

    # Or:
    # g.write_graph('example.graphml')

    # Or, to pretty-print with whitespace:
    # g.write_graph('pretty_example.graphml', pretty_print=True)


# -

def spacydoc2pyyed(sdoc: spacy.tokens.Doc):
    # Valid node shapes are: "rectangle", "rectangle3d", "roundrectangle", "diamond", "ellipse", "fatarrow", "fatarrow2", "hexagon", "octagon", "parallelogram", "parallelogram2", "star5", "star6", "star6", "star8", "trapezoid", "trapezoid2", "triangle", "trapezoid2", "triangle"
    # Valid line_types are: "line", "dashed", "dotted", "dashed_dotted"
    # Valid font_styles are: "plain", "bold", "italic", "bolditalic"
    # Valid arrow_types are: "none", "standard", "white_delta", "diamond", "white_diamond", "short", "plain", "concave", "concave", "convex", "circle", "transparent_circle", "dash", "skewed_dash", "t_shape", "crows_foot_one_mandatory", "crows_foot_many_mandatory", "crows_foot_many_optional", "crows_foot_many_optional", "crows_foot_one", "crows_foot_many", "crows_foot_optional"
    import pyyed
    g = pyyed.Graph()
    # tok :  spacy.tokens.token.Token
    for tok in sdoc:
        color = "#ffcc00"
        if tok.pos_ == "NOUN":
            color = "#00ccff"
        elif tok.pos_ == "VERB":
            color = "#00ff00"
        if tok.ent_type_:
            color = "#ff0000"

        g.add_node(tok.i, label=tok.text, shape_fill=color)

    for tok in sdoc:
        if not tok.head.i == tok.i:
            g.add_edge(tok.head.i, tok.i, label="EDGE!", width="1.0", color="#000000",
                       arrowhead="standard", arrowfoot="none", line_type="line")

    save_to_file(g)


# save graphml from networkx:
# nx.write_graphml_lxml(G,'test.graphml')
# nx.write_graphml(G,'test.graphml')


def graphviz_node_style(
        style='filled', fillcolor="white", shape="box",
        fontsize=8, width=0.01, height=0.01
):
    """Default style attributes for graphviz nodes  as a convenience function"""
    return dict(
        style=style, fillcolor=fillcolor, shape=shape,
        fontsize=fontsize, width=width, height=height
    )


class NodeFlags(str, Enum):
    """Some flags for nodes which can be used to indicate styles"""
    entity = 'entity'
    relation = 'relation'


def simple_graphviz_styling(G):
    G = G.copy()
    for n in (G.nodes[ni] for ni in G.nodes):
        n.update(graphviz_node_style())

    for e in (G.edges[ei] for ei in G.edges):
        e.update(dir="forward", color="#000000",
                 arrowhead="normal", arrowtail="none", line_type="line",
                 fontsize=8)
    return G


def graphviz_styling(G):
    G = G.copy()
    for n in (G.nodes[ni] for ni in G.nodes):
        logger.debug(f"style node: {n}")
        color = "#ffcc00"
        shape = "box"
        flags = n.get('flags', [])
        token = n.get('tok_range', [])
        if len(token) == 1:
            tok = token[0]
            if tok.pos_ == "NOUN":
                color = "#00ccff"
            elif tok.pos_ == "VERB":
                color = "#00ff00"
            if tok.ent_type_:
                color = "#ff0000"
        else:
            color = "#00ccff"
            shape = "octagon"

        if mods := n.get('modifiers', None):
            n['label'] += f"\n{mods}"

        if "entity" in flags:
            color = "#ff0000"
        elif "relation" in flags:
            shape = "cbs"
        n.update(graphviz_node_style(fillcolor=color, shape=shape))

    for e in (G.edges[ei] for ei in G.edges):
        if (label := e.get('relation')) != 'relation':
            label = f"{e['relation']}\n{spacy.explain(e['relation'])}"
        # label = spacy.explain(tok.tag_)
        e.update(label=label, dir="forward", color="#000000",
                 arrowhead="normal", arrowtail="none", line_type="line",
                 fontsize=8)
    return G


def draw(graph, engine="dot", format='jpg'):
    """format can also be "svg" and more suppoted by graphviz"""
    graphviz = nx.nx_agraph.to_agraph(graph)
    graphviz.graph_attr["overlap"] = "false"
    res = graphviz.draw(prog=engine, format=format)
    if format == 'svg':
        # this is for ipython notebooks
        # svg_html = '<div style="width:500%; height:500%;">{}</div>'.format(svg)
        # display(HTML(svg_html))
        return SVG(res)
    else:
        return res


def draw_knowledge_graph(KG):
    import matplotlib.pyplot as plt
    pos = nx.spring_layout(KG, seed=42)
    plt.figure(figsize=(12, 12))

    nx.draw(KG, with_labels=True, node_color='skyblue', edge_cmap=plt.cm.Blues, pos=pos)
    edge_labels = nx.get_edge_attributes(KG, 'type')
    nx.draw_networkx_edge_labels(KG, pos, edge_labels=edge_labels)

    plt.show()
