#!/usr/bin/env python

import requests
from bs4 import BeautifulSoup


def get_soup(url:str):
    """Get soup from an url
    """
    resp = requests.get(url)
    html = resp.content
    return BeautifulSoup(html, features='html.parser')


class Fandom:

    def __init__(self, name:str):
        self.url=f"https://{name}.fandom.com/wiki/"
        self.data=None


    def filter_valid_planet_name(self, planet_name:str):
        """Filter unvalid planet names from planets index page.
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
        """
        """
        if debug:
            letters = list(map(chr, range(97, 123)))
        else:
            letters = ['T']
        planet_list=[]
        for letter in letters:
            soup = get_soup(alpha_page + letter)
            planets=soup.find_all('a', class_='category-page__member-link')
            for name in planets:
                name = name.text
                if self.filter_valid_planet_name(name):
                    planet_list += [name]
        return planet_list


    def parse_planet_page(self, url:str):
        """Parser for planet page within a fandom website.
        """

        soup = get_soup(url)

        data_label = soup.find_all('h3', class_='pi-data-label')
        if len(data_label)<2:
            return None
        data_label = [_.text for _ in data_label]

        data_value = soup.find_all('div', class_='pi-data-value')
        data_value = [_.text for _ in data_value]

        data = {key:val for key,val in zip(data_label, data_value)}

        data_item = soup.find_all('h2', class_='pi-item')[0].text \
            if len(soup.find_all('h2', class_='pi-item')) > 0 else None
        data['pl_name'] = data_item

        return data


    def load_data(self, debug=False):
        """load data
        """
        alpha_pages_query_url = self.url + "Category:Planets?from="
        planet_names = self.extract_planet_names(alpha_pages_query_url, debug=debug)

        data = []
        for name in planet_names:
            url_planet = f"{self.url}{name.replace(' ', '_')}"
            if debug:
                print(url_planet)
            data += [self.parse_planet_page(url_planet)]

        return data


if __name__ == "__main__":

    # mass effect
    # fandom = Fandom("masseffect")

    # alien vs predator
    fandom = Fandom("avp")

    # load data
    data=fandom.load_data(debug=True)
    print(data[0])
