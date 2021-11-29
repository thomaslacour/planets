#!/usr/bin/env python

import utils

class masseffect(utils.fandom)
    full_url='https://avp.fandom.com/wiki/Category:Planets'

def load_data():
    fandom = utils.Fandom("avp")
    data = fandom.load_data(debug=True)
    return data
