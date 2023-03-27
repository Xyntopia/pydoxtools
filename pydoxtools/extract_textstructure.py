import operator
import typing
from dataclasses import asdict

import pandas as pd
import pdfminer
from pdfminer.layout import LTChar, LTTextLineVertical
from sklearn.ensemble import IsolationForest

from pydoxtools import document_base


def _line2txt(LTOBJ: typing.Iterable):
    """
    extract text from pdfiner.six lineobj including size hints

    TODO: speedup using cython/nuitka/numba
    """
    txt = ""
    last_size = 0
    for i, ch in enumerate(LTOBJ):
        newtxt = ""
        sizehint = ""
        if isinstance(ch, pdfminer.layout.LTText):
            newtxt = ch.get_text()
        if isinstance(ch, pdfminer.layout.LTChar):
            newsize = ch.size
            if i > 0:
                # TODO: use an iterative function here...
                if newsize < last_size:
                    sizehint = "<s>"
                elif newsize > last_size:
                    sizehint = "</s>"
            last_size = newsize
        txt += sizehint + newtxt
    return txt


def docinfo(self) -> list[dict[str, str]]:
    """list of document metadata such as author, creation date, organization"""
    return []


def num_pages(self) -> int:
    return len(self.pages)


def pages(self) -> list[str]:
    """automatically divide text into approx. pages"""
    page_word_size = 500
    words = self.full_text.split()
    # for i in range(len(words)):
    pages = list(words[i:i + page_word_size] for i in range(0, len(words), page_word_size))
    return pages


def mime_type(self) -> str:
    """
    type such as "pdf", "html" etc...  can also be the mimetype!
    TODO: maybe we can do something generic here?
    """
    return "unknown"


class DocumentElementFilter(document_base.Extractor):
    """Filter document elements for various criteria"""

    def __init__(self, element_type: document_base.ElementType):
        super().__init__()
        self.element_type = element_type

    def __call__(self, elements: pd.DataFrame):
        df = elements.loc[elements["type"] == self.element_type].copy()
        return df


class TextBoxElementExtractor(document_base.Extractor):
    """
    create textboxes and create bounding boxes and aggregated text from
    a pandas dataframe with textlines.
    returns a list of textboxes together wth some coordinate data and
    the contained text.

    TODO: generalize this function into an "aggregator" function

    TODO: make it possible to use alternative distance metrics to generate
          the text boxes...

    TODO: detect textboxes if they weren't loaded from another framewok
          already (for example pdfminer.six automatically detects textboxes ad
          we save them in the elements array)

    TODO: do some schema validation on the pandas dataframes...
    """

    def __call__(self, line_elements: pd.DataFrame):
        if "boxnum" in line_elements:
            group = line_elements.groupby(['p_num', 'boxnum'])
            # aggregate object from the same box and calculate new
            # bounding boxes, also join the formatted text
            bg = group.agg(
                x0=("x0", "min"), y0=("y0", "min"),
                x1=("x1", "max"), y1=("y1", "max"),
                text=("lineobj",
                      lambda x: "".join(_line2txt(obj) for obj in x.values))
            )
            # remove empty box_groups
            bg = bg[bg.text.str.strip().str.len() > 1].copy()
            # do some calculations
            bg['y_mean'] = bg[['y0', 'y1']].mean(axis=1)
            bg['x_mean'] = bg[['x0', 'x1']].mean(axis=1)
            bg['w'] = bg.x1 - bg.x0
            bg['h'] = bg.y1 - bg.y0
            return dict(text_box_elements=bg)
        else:
            return dict(text_box_elements=None)


class TitleExtractor(document_base.Extractor):
    """
    This Extractor extracts titels and other interesting text parts
    from a visual document. It does this by characterising parts
    of the text being "different" than the rest using an
    Isolation Forest algorithm (anomyla detection).
    Features are for example: font size,
    position, length etc...

    #TODO: use this for html and other kinds of text files as well...
    """

    def __call__(self, line_elements):
        dfl = self.prepare_features(line_elements)
        return dict(
            titles=self.titles(dfl),
            side_titles=self.side_titles(dfl),
            side_content=self.side_content(dfl),
            main_content=self.normal_content(dfl)
        )

    def prepare_features(self, df_le: pd.DataFrame) -> pd.DataFrame:
        """
        detects titles and interesting textpieces from a list of text lines
        TODO: convert this function into a function of "feature-generation"
              and move the anomaly detection into the cached_property functions
        """

        # TODO: extract the necessary features that we need here "on-the-fly" from
        #       LTLineObj
        # extract more features for every line
        dfl = df_le
        # get font with largest size to characterize line
        # TODO: this can probably be made better..  (e.g. only take the font of the "majority" content)
        dfl[['font', 'size', 'color']] = dfl.font_infos.apply(
            lambda x: pd.Series(asdict(max(x, key=operator.attrgetter("size"))))
        )

        # generate some more features
        dfl['text'] = dfl.rawtext.str.strip()
        dfl = dfl.loc[dfl.text.str.len() > 0].copy()
        dfl['length'] = dfl.text.str.len()
        dfl['wordcount'] = dfl.text.str.split().apply(len)
        dfl['vertical'] = dfl.lineobj.apply(lambda x: isinstance(x, LTTextLineVertical))

        dfl = dfl.join(pd.get_dummies(dfl.font, prefix="font"))
        dfl = dfl.join(pd.get_dummies(dfl.font, prefix="color"))

        features = set(dfl.columns) - {'gobj', 'linewidth', 'non_stroking_color', 'stroking_color', 'stroke',
                                       'fill', 'evenodd', 'type', 'text', 'font_infos', 'font', 'rawtext', 'lineobj',
                                       'color'}

        # detect outliers to isolate titles and other content from "normal"
        # content
        # TODO: this could be subject to some hyperparameter optimization...
        df = dfl[list(features)]
        clf = IsolationForest()  # contamination=0.05)
        clf.fit(df)
        dfl['outliers'] = clf.predict(df)

        return dfl

    def titles(self, dfl) -> typing.List:
        # titles = l.query("outliers==-1")
        titles = dfl.query("outliers==-1 and wordcount<10")
        titles = titles[titles['size'] >= titles['size'].quantile(0.75)]
        return titles.get("text", pd.Series(dtype=object)).to_list()

    def side_titles(self, dfl) -> pd.DataFrame:
        # TODO: what to do with side-titles?
        side_titles = dfl.query("outliers==-1 and wordcount<10")
        side_titles = side_titles[side_titles['size'] > dfl['size'].quantile(0.75)]
        # titles = titles[titles['size']>titles['size'].quantile(0.75)]
        return side_titles

    def side_content(self, dfl) -> str:
        # TODO: extract side-content such as addresses etc..
        side_content = "\n---\n".join(dfl[dfl.outliers == -1].text)
        return side_content

    def normal_content(self, dfl) -> str:
        # TODO: what does this function do, I forgot...
        main_content = "\n---\n".join(dfl[dfl.outliers == 1].text)
        return main_content
