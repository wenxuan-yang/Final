"""
Yutian Lei, Wenxuan Yang, Yining Liu
CSE163
This class builds a data reader object that help
our project for data collection by applying
requests and pandas libraries
"""


import requests

import pandas as pd


class DataReader:
    """
    Represents a data reader with web data scraping
    function
    """

    def __init__(self, _url):
        """
        Initializes a new data reader with the given url
        """
        self._url = _url

    def read(self):
        """
        Returns a DataFrame and a list with the data
        obtained from the website of given url.
        :return: The DataFrame containing the numeric
                data we need from the web.
                The list contains the column names
                of the data we collected from web.
        """
        _paper = requests.get(self._url).text
        _lines = _paper.splitlines()
        _generate_category = False
        _record_data = False
        _categories = []
        _data = []
        for each in _lines:
            line = each.strip()
            if len(line) != 0:
                if line == 'The variables listed below, from left to right, ' \
                           'are:':
                    _generate_category = True
                elif line == '(Measurement standards are apparently those ' \
                             'listed in Benhke and Wilmore':
                    _generate_category = False
                elif _generate_category:
                    _categories.append(line)
                if line == 'Principles of the Conditioning Process_, Allyn ' \
                           'and Bacon, Inc., Boston.':
                    _record_data = True
                elif line == 'Roger W. Johnson':
                    _record_data = False
                elif _record_data:
                    _data.append(line.split())
        _processed_data = pd.DataFrame(_data, columns=_categories)
        _processed_data = _processed_data.astype(float)
        return _processed_data, _categories
