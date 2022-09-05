from document import Extractor
from typing import Any


class TableExtractor(Extractor):
    @property
    def list_lines(self):
        return []

    @property
    def tables(self) -> list[dict[str, dict[str, Any]]]:
        """
        table in the following (row - wise) format:

        [{index -> {column -> value } }]
        """
        return []

    @property
    def tables_df(self) -> List["pd.DataFrame"]:
        return []
