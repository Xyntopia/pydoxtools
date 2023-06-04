from __future__ import annotations  # this is so, that we can use python3.10 annotations..

import base64
import html
import logging
from io import BytesIO
from typing import List
from urllib.parse import urlsplit

import extruct
import langdetect.lang_detect_exception
import lxml
import numpy as np
import pandas as pd
import readability
from bs4 import BeautifulSoup, NavigableString
from goose3 import Goose

import pydoxtools.operators_base
from pydoxtools import html_utils
from pydoxtools.html_utils import logger, clean_html

try:
    # import a couple libraries that are only
    # needed for analysis purposes
    import selenium
    from selenium import webdriver
    import selenium.common.exceptions
    from PIL import Image
except ImportError:
    logger.info("can not use selenium, due to missing libraries: selenium, PIL")
    selenium = None

logger = logging.getLogger(__name__)


def find_language(html):
    txt = html_utils.get_pure_html_text(html)
    try:
        return langdetect.detect(txt)
    except langdetect.lang_detect_exception.LangDetectException:
        return "None"


def extract_tables(html):
    """
    tries to extract all tables from an html using pandas dataframe

    TODO: similar to extract_lists try to only extract
        tables which seem to contain "valid" information. for example
        which only have a maximum depth of the DOM tree
        or which have not "only links".

    TODO: extract div-lists & tables
    """
    try:
        new_tables = pd.read_html(html, header=None)
    except ValueError:
        logger.debug(f"no tables found")
        new_tables = []
    except lxml.etree.ParserError:
        # probably an empty document
        new_tables = []
    return new_tables


def goose_extract(raw_html: str):
    goose = Goose()
    article = goose.extract(raw_html=raw_html)
    main_content = str(article.cleaned_text)
    if main_content != '':
        # TODO: maybe we should use html_utils.strip_html for this task?
        html_raw = str(lxml.etree.tostring(article.top_node, pretty_print=True))
        main_content_clean_html = clean_html(html_raw)
    else:
        main_content_clean_html = ''
    keywords = article.meta_keywords
    summary = str(article.meta_description)
    language = str(article.meta_lang)
    # more data:
    # article.infos
    return main_content_clean_html, main_content, keywords, summary, language, article


def extract_title_and_content(raw_html):
    """this function takes the html
    and extracts the important parts of it"""
    doc = readability.Document(raw_html)
    return pd.Series((doc.summary(), doc.title(), doc.short_title()))


# the folowing dfines a couple "forbidden" tags which are not accepted
# for specification-lists to be valid. They make sure we only select
# list with valid information about a component.
# - "a" - tag because it is a link
# - "button" because it also represents a link/navigation element
# - svg becuase most often it represents "payment" buttons and other links
_forbidden_list = ['a', 'button', 'svg', 'form', 'input']


def is_link_only(el):
    """
    This function finds out if the DOM tree contains only link-branches or also has other content.

    this means if any element in this tree/chain:

    ul -> li -> a -> p -> div -> "Text" is a link only, it will return "True"

    a tree like this:

    ul -> li -> "sometext"
             |-> a -> p -> "sometext"

    will return "False" as there is a branch which does not contain a link element

    while this one:

    ul -> li -> a ...
             |-> a ...
             |-> a ...

    will also return True, as it contains only link-branches
    """
    # children = el.children#(True, recursive=False)
    children = el.contents
    # print(f"inside {el.name}, children#: {len(children)}")
    # print(f"# of chilren: {len(children)}")
    # only if there is "valid" content, return False,
    # signaling that this might not be a link-only-element
    if len(children) == 0 and el.text.strip():
        # print("no more children, returning False")
        return False
    for i in children:
        # print(f"{i.name} encountered")
        if isinstance(i, NavigableString):
            # as the element doesn't have a name its probably some text ...
            if i.string.strip():
                # print("encountered non-empty text element")
                return False  # if text is not empty
            else:
                return True
        if i.name:
            if i.name not in _forbidden_list:
                # print(f"diving into {i.name}")
                if not is_link_only(i): return False
    return True


def clean_html_list(l):
    """
    filter lists for lists that do not contain only links.
    As those hold the important information.
    """
    l = [i.text.strip() for i in l if not is_link_only(i)]
    return l


def extract_lists(html: str) -> List:
    """
    searches for all kinds of lists inside of html code

    right now, everything with with "<ul>" tags.

    The function will throw out list items which contain only links
    and no other text in order to filter out navigation elements
    from webpages.

    TODO: search for "similar" list items. Meaning: if we have
          a row of "similar" looking divs, there is a high probability
          of having a list. Might be best to do this using a machine learning
          algorithm.
    """
    if html is None:
        return []
    soup = BeautifulSoup(html, "lxml")
    lists = soup('ul')
    # TODO: list_headers = [li.find_previous_sibling() for li in lists]
    # remove all "newlines" from lists
    ls = [[i for i in li.children if i.name == 'li'] for li in lists]
    ls = [clean_html_list(li) for li in ls]  # remove list items with only links
    ls = [[j for j in li if j] for li in ls]  # remove empty list items
    ls = [i for i in ls if i]  # remove empty lists
    return ls


def extract_schema(raw_html, url):
    baseurl = urlsplit(url).netloc
    # all meta tags
    data = extruct.extract(raw_html, baseurl, uniform=True)
    return data


class HtmlExtractor(pydoxtools.operators_base.Operator):
    def __init__(self, engine="combined"):
        super().__init__()
        self._engine = engine

    def __call__(self, raw_html: str, url: str = ""):
        url = str(url)

        if self._engine == "combined":
            try:  # try to unescape html its ok if it doesn't work
                raw_html = html.unescape(raw_html)
            except TypeError:
                pass
            main_content_html2, title, short_title = extract_title_and_content(raw_html)

            (main_content_clean_html, main_content,
             keywords, summary, language, article) = goose_extract(raw_html)

            # if we didn't find a given language, try to detect on our own...
            if language.lower() == 'none':
                language = find_language(main_content)

            lists = extract_lists(raw_html)
            lists = [pd.DataFrame(l) for l in lists]
            html_tables = []
            for html_con in [main_content_clean_html, main_content_html2]:
                html_tables.extend(extract_tables(html_con))

            # add lists to tables
            tables = lists + html_tables
            # convert tables into json readable format
            schemadata = extract_schema(raw_html, url or "")

            L = [article.final_url, article.canonical_link]
            final_urls = [x for x in L if x is not None]
            # TODO: sort & order schemadata into the relevant fields in a better way
            # TODO: extract titles from html (searching for headings, etc...)

            pdf_links = html_utils.get_pdf_links(raw_html)

            # TODO: gather a list of thing sthrough regular expressions
            # product_ids = []
            # prices
            # regex_ex = {}

        else:
            raise NotImplementedError()

        # TODO: decide which main content is the "correct one"
        # TODO: deduplicate tables
        # TODO: move table detection into its own class
        # TODO: move moe functions into their own Extractors
        return dict(
            main_content_clean_html=main_content_clean_html,
            main_content=main_content,
            html_keywords=keywords,
            summary=summary,
            language=language,
            goose_article=article,
            tables=tables,
            schemadata=schemadata,
            final_urls=final_urls,
            pdf_links=pdf_links,
            title=title,
            short_title=short_title,
            url=url
        )


class SeleniumDriver:
    """
    onwards from ubuntu 20.04, chromium-browsre can only be installed
    through snap packages. This prevents access to any folder outside the home directory
    which is problematic if we want to open files from /tmp for example.
    We therefore have to start chrome with a couple extra-options. And directly the snap...
    """

    def __init__(self, headless=True):
        if selenium is None:
            raise ImportError("Selenium is probably not installed...")
        browser = "chromium"
        if browser == "chromium":  # chromium/chrome
            # noinspection PyUnresolvedReferences
            options = selenium.webdriver.chrome.options.Options()
            # options.binary_location = "/usr/bin/google-chrome-stable"
            options.binary_location = "/snap/chromium/current/usr/lib/chromium-browser/chrome"
            if headless: options.add_argument("--headless")
            if False: "--no-startup-window"
            # chrome_options.add_argument("--test-type")
            options.add_argument("--no-sandxbox")  # this is needed to run as root in docker
            options.add_argument('--ignore-certificate-errors')
            options.add_argument('--disable-gpu')
            options.add_argument('--window-size=1280,1000')
            options.add_argument('--allow-insecure-localhost')
            options.add_argument('--allow-running-insecure-content')
            options.add_argument('--remote-debugging-port=9222')
            options.add_argument('--allow-file-access-from-files')
        elif browser == "firefox":
            options = webdriver.FirefoxOptions()
            options.add_argument("-headless")

        try:
            if browser == "chromium":
                self.driver = webdriver.Chrome(
                    '/usr/lib/chromium-browser/chromedriver',
                    options=options)
            elif browser == "firefox":
                # /usr/lib/firefoxdriver/webdriver.xpi
                # /usr/share/lintian/overrides/firefoxdriver
                # /usr/lib/firefoxdriver/amd64/x_ignore_nofocus.so
                # /usr/lib/firefoxdriver/x86/x_ignore_nofocus.so

                self.driver = webdriver.Firefox(
                    '/usr/share/lintian/overrides/firefoxdriver',
                    options=options)
        except selenium.common.exceptions.WebDriverException:
            logger.exception("""
            
            Chrome Driver is probably no installed, to install it do the following:
            
            sudo apt install chromium-chromedriver
            """)

    def url2image(self, url):
        self.driver.get(url)
        png = self.driver.get_screenshot_as_png()
        im = Image.open(BytesIO(png))
        return im

    def openHTML(self, htmlString):
        html_bs64 = base64.b64encode(htmlString.encode('utf-8')).decode()
        try:
            self.driver.get("data:text/html;base64," + html_bs64)
            return True
        except selenium.common.exceptions.TimeoutException:
            return False

    def urlOpen(self, url):
        try:
            logger.info(f"opening {url}")
            self.driver.get(url)
            logger.info(self.driver.current_url)
            return True
        except:
            self.driver.delete_all_cookies()
            logger.warning(f"timeout opening {url}")
            return False

    def get_page_source(self):
        # self.urlOpen(url)
        return self.driver.page_source

    def get_page_image(self):
        logger.info("image successfully opened!")
        png = self.driver.get_screenshot_as_png()
        im = Image.open(BytesIO(png))
        # driver.quit()
        return im

    def html2image(self, html):
        """
        Takes a screenshot by first saving the html as a temporary file
        and then opening it in selenium chromium driver.

        Parameters
        ----------
        html : TYPE
            DESCRIPTION.

        Returns
        -------
        im : TYPE
            DESCRIPTION.

        """

        self.driver.set_page_load_timeout(10)  # Set timeout of n seconds for page load

        success = self.openHTML(html)
        if not success:
            return Image.fromarray(
                (np.random.rand(50, 50, 3) * 255).astype('uint8'),
                'RGB'
            )
        else:
            logger.debug("image successfully opened!")
            png = self.driver.get_screenshot_as_png()
            im = Image.open(BytesIO(png))
            # driver.quit()
            return im

    def close(self):
        self.driver.close()
        self.driver.quit()


def get_main_content(html):
    """TODO: try to merge this function somehow :P"""
    return readability.Document(html).summary()
