#!/usr/bin/env python

import pandas as pd
import requests
from bs4 import BeautifulSoup
from planets.scrapper import utils
from planets.scrapper.utils import Scrapper, TableScrapper


class Wikipedia(Scrapper):

    table_class="wikitable"

    def __init__(self, url:str=f"https://en.wikipedia.org/wiki/%?", page="<pagename>" ):
        super().__init__(url.replace('%?', page))
        self.data=None

    def scrap_table(self, debug:bool=False):
        """
        """
        soup = self.get_soup(self.url)
        table = soup.find_all('table', {'class':Wikipedia.table_class})
        return pd.read_html(str(table))
