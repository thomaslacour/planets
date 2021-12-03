#!/usr/bin/env python

import requests
from bs4 import BeautifulSoup
import pandas as pd

class Scrapper:
    """Scrapper class mold for scrapping
    """

    def __init__(self, url:str):
        self.url=url
        self.data=None


    def get_soup(self, url:str) -> BeautifulSoup:
        """Get soup from an url
        """
        resp = requests.get(url)
        html = resp.content
        return BeautifulSoup(html, features='html.parser')

    def scrap(self, debug:bool=False):
        return None



class TableScrapper(Scrapper):
    """Scrapper class for tables
    """
    def __init__(self, url, table_class=None):
        super().__init__(url)
        self.table_class=table_class

    def scrap(self):
        if not self.table_class:
            return pd.read_html(self.url)
        soup = self.get_soup(self.url)
        table = soup.find_all('table', {'class':self.table_class})
        return pd.read_html(str(table), flavor='bs4', thousands=' ', decimal=',')



def transform_to_df(data:list) -> pd.DataFrame:
    return pd.DataFrame(data)
