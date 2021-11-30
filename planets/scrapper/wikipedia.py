#!/usr/bin/env python

import pandas as pd
import requests
from bs4 import BeautifulSoup
from planets.scrapper import utils
from planets.scrapper.utils import Scrapper, TableScrapper


class Wikipedia(Scrapper):

    html_table_class="wikitable"

    def __init__(self, url:str=f"https://en.wikipedia.org/wiki/%?", page:str="<pagename>" ):
        super().__init__(url.replace('%?', page))
        self.data=None

    def scrap_table(self, debug:bool=False) -> pd.DataFrame:
        """Table scrapper.
        """
        soup = self.get_soup(self.url)
        table = soup.find_all('table', {'class':Wikipedia.html_table_class})
        return pd.read_html(str(table))
