#!/usr/bin/env python

import utils

class masseffect(utils.fandom)
    full_url='https://masseffect.fandom.com/wiki/Category:Planets'

def load_data():
    fandom = utils.Fandom("masseffect")
    data = fandom.load_data(debug=True)
    return data
