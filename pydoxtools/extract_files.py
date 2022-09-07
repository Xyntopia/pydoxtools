from pydoxtools import document


class txtExtractor(document.Extractor):
    def __call__(self, fobj, page_numbers=None, max_pages=None):
        if isinstance(fobj, str):
            txt = fobj
        else:
            with open(fobj) as file:
                txt = file.read()
        return dict(txt=txt)
