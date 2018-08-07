#!/usr/bin/env ipython
import json
def load_settings_from_file(identifier):
    """
    Handle loading settings from a JSON file, filling in missing settings from
    the command line defaults, but otherwise overwriting them.
    """
    settings_path = './experiments/settings/' + identifier + '.txt'
    print('Loading settings from', settings_path)
    settings_loaded = json.load(open(settings_path, 'r'))
    # check for settings missing in file
    return settings_loaded
