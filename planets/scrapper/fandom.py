#!/usr/bin/env python

import pandas as pd
import requests
from bs4 import BeautifulSoup
from planets.scrapper import utils
from planets.scrapper.utils import Scrapper

class Fandom(Scrapper):

    def __init__(self, url:str=f"https://%?.fandom.com/wiki/", page:str="<pagename>" ):
        super().__init__(url.replace('%?', page))
        self.data=None


    def is_valid_planet_name(self, planet_name:str) -> bool:
        """Filter invalid planet names from planets index page.
        """
        planet_name=planet_name.lower()
        if 'template' in planet_name:
            return False
        if 'wiki' in planet_name:
            return False
        if planet_name.endswith(('.png', '.jpg', '.jpeg')):
            return False
        return True


    def extract_planet_names(self, alpha_page:str, debug:bool=False) -> dict:
        """Extract planet names from the index alpha page of planet wiki.
        """
        # create a list of all letters or test only for one if debug
        if not debug:
            letters = list(map(chr, range(97, 123)))
        else:
            letters = ['T']

        planet_list=[]
        for letter in letters:
            soup = self.get_soup(alpha_page + letter)
            planets = soup.find_all('a', class_='category-page__member-link')
            for name in planets:
                name = name.text
                if self.is_valid_planet_name(name):
                    planet_list += [name]
        return planet_list


    def parse_planet_page(self, url:str) -> dict:
        """Parser for planet page within a fandom website.
        """

        # get html soup
        soup = self.get_soup(url)

        # get the label of collected data
        data_label = soup.find_all('h3', class_='pi-data-label')
        if len(data_label)<2:
            return {'pl_name': None}
        data_label = [_.text for _ in data_label]

        # get the value of collected data
        data_value = soup.find_all('div', class_='pi-data-value')
        data_value = [_.text for _ in data_value]

        # convert data to a dictionary
        data = {key:val for key,val in zip(data_label, data_value)}

        # retrieve planet name
        data_item = soup.find_all('h2', class_='pi-item')[0].text \
            if len(soup.find_all('h2', class_='pi-item')) > 0 else None
        data['pl_name'] = data_item

        return data


    def scrap(self, debug:bool=False) -> pd.DataFrame:
        """
        """
        # url to query the alpha page that listed planets order by names
        alpha_pages_query_url = self.url + "Category:Planets?from="

        # get list of planet names
        planet_names = self.extract_planet_names(alpha_pages_query_url, debug=debug)

        # scrapping for every pages associated to the list of planet names
        data = []
        for name in planet_names:
            url_planet = f"{self.url}{name.replace(' ', '_')}"
            data += [self.parse_planet_page(url_planet)]
            if debug:
                print(url_planet, '\n', data[-1], '\n--------------')

        # conversion to a DataFrame
        self.data = utils.transform_to_df(data)

        return self.data


if __name__ == "__main__":

    # alien vs predator fandom wiki
    fandom = Fandom("avp")
    print(fandom.url)

    # load data
    data=fandom.load_data(debug=True)

