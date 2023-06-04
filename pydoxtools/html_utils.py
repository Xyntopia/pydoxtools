# !/usr/bin/env python3
"""
Created on Mon Apr  6 21:57:32 2020
# TODO write file description
# TODO split up file in "cleaning" and "extraction" functions
"""
from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import logging
import os
import re
import tempfile
import urllib
import webbrowser
from urllib.parse import urlparse

import pandas as pd
from bs4 import BeautifulSoup
from lxml.html.clean import Cleaner

logger = logging.getLogger(__name__)


def bs(html):
    return BeautifulSoup(html, 'lxml')


# extract tables
def save_tmp_html(html: str):
    with tempfile.NamedTemporaryFile('w+', delete=False,
                                     suffix='.html', encoding='utf-8') as f:
        url = 'file://' + f.name
        f.write(html)
    return url


def open_in_browser(data_object: str):
    if data_object.startswith("http://"):
        webbrowser.open(data_object)
    elif isinstance(data_object, str):
        url = save_tmp_html(data_object)
        webbrowser.open(url)


def open_in_browser2(data_object):
    if isinstance(data_object, pd.core.series.Series):
        os.system(f"chromium-browser {data_object.url}")
    else:
        os.system(f"chromium-browser {data_object}")


def markdownify(html):
    NotImplementedError()
    from markdownify import markdownify
    h2 = strip_html(html)
    return markdownify(h2)


def strip_html(html):
    """
    create a very clean, but still html version of the
    samepage. This should roughly retain the structure of the content.

    https://lxml.de/api/lxml.html.clean.Cleaner-class.html

    # TODO: use this in extract_html for main content?
    """
    cleaner = Cleaner(page_structure=False,
                      meta=True,
                      embedded=True,
                      links=True,
                      style=True,
                      processing_instructions=True,
                      inline_style=True,
                      scripts=True,
                      javascript=True,
                      comments=True,
                      frames=True,
                      forms=True,
                      annoying_tags=True,
                      remove_unknown_tags=True,
                      safe_attrs_only=False,
                      safe_attrs=frozenset(['src', 'color', 'href', 'title', 'class', 'name', 'id']),
                      remove_tags=('span', 'font', 'div', 'img')
                      )

    return cleaner.clean_html(html)


def get_text_only_blocks(html) -> list[str]:
    """
    extrac html text, remove most whitespace and split into textblock lists
    """
    return re.split(r"\s*\n\s*\n\s*", bs(html).get_text("\n\n").strip())


def get_pure_html_text(html):
    " ".join(get_text_only_blocks(html))
    return txt.strip()


def url_join(url1, url2):
    """
    joins two urls while removing double-slashes
    """
    # TODO: remove double slashes from URL with http
    url1, url2 = (url.replace("//", "/") if url[:4] != "http"
                  else url[:7] + url[7:].replace("//", "/") for url in [url1, url2])
    return urllib.parse.urljoin(url1, url2)


def get_pdf_links(html) -> list[str]:
    linklist = []
    soup = BeautifulSoup(html, 'lxml')
    for link in soup.find_all('a', href=True):
        if link['href'].lower().endswith(".pdf"):
            linklist.append(link['href'])
    return linklist


def absolute_url_path(url: str, path: str) -> str:
    if path.startswith("http"):
        return path
    else:
        return urlparse(url)._replace(path=path).geturl()


def clean_html(html):
    # TODO do this in a better way ;)
    # it gets rid of several tags  in an html
    soup = BeautifulSoup(html, "lxml")  # , features="html.parser")
    for tag in soup():
        for attribute in ["class", "id", "name", "style"]:
            del tag[attribute]
    return str(soup)


def prettyprint(html):
    soup = BeautifulSoup(html, "lxml")  # make BeautifulSoup
    return soup.prettify()  # prettify the html


def color_text(txttok, txtval):
    """
    generate html page with colored text tokens "txttok" according
    to a list "txtval".
    """
    import yattag
    txtval_norm = [(i - min(txtval)) / (max(txtval) - min(txtval))
                   for i in txtval]

    doc, tag, text = yattag.Doc().tagtext()
    with tag('h1'):
        for tok, val in zip(txttok, txtval_norm):
            cv = int((1 - val) * 255)
            col = f"rgb(255,{cv}, {cv})"
            with tag('span', style=f"background-color: {col};"):
                text(f"{tok} ")

    return doc.getvalue()


def text_density(soup):
    return len(soup.get_text()) / len(str(soup))


def text_density_stripped(soup):
    raise NotImplementedError()
    b = ' '.join(a.get_text().split())
    c = ' '.join(str(a).split())
    return len(b) / len(c)


def update_listdict(items, newitems):
    for key, val in newitems.items():
        x = items.get(key, [])
        if isinstance(val, list):
            x.extend(val)
        else:
            x.append(val)
        items[key] = x
    return items


if __name__ == "__main__":
    import sys

    sys.path.append("..")

    import pydoxtools.training

    ldata = pydoxtools.training.load_labeled_webpages()

    ldata['len'] = ldata.raw_html.str.len()

    page = ldata.loc[ldata.len.idxmax()]
    del ldata

    html = page.raw_html
    html

    len(html)
    html3 = ' '.join(get_pure_html_text(html).split())
    html3p = prettyprint(html3)
    len(html3)

    html4 = bs(html).get_text(strip=True)
    len(html4)

    if False:
        if False:
            txt = "Kurzbeschreibung \n\n Dieser Wireless LAN USB Stick kann an einen Computer angeschlossen werden, um mit einem WLAN Netzwerk verbunden zu werden. Der WLAN Stick kann mit einem AccessPoint (HotSpot) oder WLAN-Router im 2,4 GHz und 5 GHz Band verbunden werden. Des Weiteren ist eine adhoc Verbindung zu einem zweiten"

            txttok = txt.split()
            txtval = [i for i, tok in enumerate(txttok)]

            html = color_text(txttok, txtval)

            oib(html)
            print(html)

        dr = get_selenium_driver()
        im = html2image(html, dr)
        dr.quit()
