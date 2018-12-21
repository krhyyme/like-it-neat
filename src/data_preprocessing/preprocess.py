#!/usr/bin/env python3

import pandas as pd
from urllib.parse import urlparse
import numpy as np


def _strip_and_lower_series(series):
    """
    A function used to strip spaces from ends of string and lowercases all characters. Replaces empty strings as numpy missing value.

    requires:
        series: series of string values

    ouput:
        out_series: series with stripped spaces and lower case characters. Replaces empty characters as np.nan
    """
    out_series = series.str.strip()
    out_series = out_series.str.lower()
    out_series = out_series.replace('', np.nan)

    return out_series


def _keep_frequent_(series, count):
    """
    A function used to identify frequently occuring values. If a value doesn't occur >= count times then it is not included in the output.

    requires:
        series: series of string values or categories
        count: minimum number of occurence a value can have

    output:
        an array of values that meet frequency threshold
    """
    v_counts = series.value_counts()
    v_counts = v_counts[v_counts >= count]

    return v_counts.index


class whisky_archive_processor:
    """
    Method to preprocess whisky_archive data set for downstream use.
    Preprocessing includes:
        * Standardizing Whisky style names
        * Pruning away whiskys with a low amount of reviews
        * Whisky review URL validation
    """

    def __init__(self):

        self.whisky_archive = None

    # data preprocessing procedure
    def process(self, file_loc):
        """
        Loads raw data to data frame and executes preprocessing steps
        file_loc: location of raw data file
        """

        # Load raw whisky archive excel file as a data frame.
        self.whisky_archive = pd.read_excel(file_loc, dtype=str)

        # convert timestamps to datetime
        # coerce error as missing value (NaT)
        self.whisky_archive['Timestamp'] = pd.to_datetime(
            self.whiskey_archive['Timestamp'], error='coerce')

        # strip spaces and lowercase characters
        self.whiskey_archive['Whisky Region or Style'] = _strip_and_lower_series(
            self.whiskey_archive['Whisky Region or Style'])

        # Alternate spellings of whisky styles are replaced and style are prioritized over location.
        replace_dict = {'bourbon/america': 'bourbon',
                        'blended speyside scotch': 'speyside',
                        'lowland / grain': 'lowland',
                        'borubon': 'bourbon',
                        'highlands': 'highland'}

        self.whiskey_archive['Whisky Region or Style'] = self.whiskey_archive['Whisky Region or Style'].replace(
            replace_dict)

        # Only include styles with at least a 100 reviews
        keep_styles = _keep_frequent_(
            series=self.whisky_archive['Whisky Region or Style'], count=100)

        # Only keep styles in keep_styles array
        self.whisky_archive = self.whisky_archive[self.whisky_archive['Whisky Region or Style'].isin(
            keep_styles)]

       # process rating data. strip spaces and lowercase charcters
       # Some users inputed there scores with a divisor. Those are being removed so only the numerator remains
       # convert ratings to float
        self.whisky_archive['Reviewer Rating'] = _strip_and_lower_series(
            self.whisky_archive['Reviewer Rating'])
        self.whiskey_archive['Reviewer Rating'] = self.whiskey_archive['Reviewer Rating'].str.replace(
            '/100', '')
        self.whiskey_archive['Reviewer Rating'] = self.whiskey_archive['Reviewer Rating'].astype(
            float)
