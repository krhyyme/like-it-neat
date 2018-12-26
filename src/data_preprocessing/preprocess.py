#!/usr/bin/env python3

import pandas as pd
from urllib.parse import urlparse
import json
import numpy as np
import praw
from praw.models import MoreComments
from tqdm import tqdm
import os


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


def _url_parse_(dataframe, url_col):
    """
    Uses urlparse to validate URLs. If urls  have their scheme, netloc, and path parsed and stored in seperate columns


    requires:
        dataframe: Pandas Dataframe containing url column
        url_col: column containing urls

    output:
        dataframe with urlparse object, scheme, netloc, and path columns

    """

    # strip and lowercase urls
    dataframe[url_col] = _strip_and_lower_series(dataframe[url_col])

    # create urlparse object
    dataframe['urlparse'] = dataframe[url_col].apply(lambda x: urlparse(x))

    # pull out attributes
    dataframe['scheme'] = dataframe['urlparse'].apply(lambda x: x.scheme)
    dataframe['netloc'] = dataframe['urlparse'].apply(lambda x: x.netloc)
    dataframe['path'] = dataframe['urlparse'].apply(lambda x: x.path)

    # replac
    for col in ['scheme', 'netloc', 'path']:
        dataframe[col] = dataframe[col].replace('', np.nan)

    return dataframe


def _url_fix_(post_url_parse_df, url_col):
    """
    Takes a dataframe with parsed URLs and corrects incorrectly parsed urls

    requires: 
        post_url_parse_df: dataframe with parsed urls
        url_col: column containing urls

    output:
       processed_dataframe:  dataframe with fixed urls
    """

    # filter out valid urls

    valid_rating_urls = post_url_parse_df[~post_url_parse_df['scheme'].isnull(
    )]
    invalid_rating_urls = post_url_parse_df[post_url_parse_df['scheme'].isnull(
    )]

    invalid_rating_urls[url_col] = invalid_rating_urls[url_col].str.replace(
        'www.', '')
    invalid_rating_urls[url_col] = 'https://www.' + \
        invalid_rating_urls[url_col]

    processed_dataframe = pd.concat([invalid_rating_urls, valid_rating_urls])

    return processed_dataframe


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


def _scrape_reddit_reviews_(dataframe, pass_loc='pass_info.json'):
    """
    This function uses the prawl library to access the reddit API and scrape the text from whisky reviell iterate through each row in the preprocessed whisky archive dataframe and pull the 
    top level comments in the review url if the comment author is the review author.
    requires:
        pass_loc: Reddit Application Data
        dataframe: Whisky Archive Dataframe
    output:
        Whisky Archive Dataframe with comment column 
    """

    # Load  authentication information
    with open('pass_info.json', 'r') as f:
        pass_info = json.load(f)

    # Intialize Reddit instance
    reddit = praw.Reddit(client_id=pass_info['client_id'],
                         client_secret=pass_info['client_secret'],
                         password=pass_info['password'],
                         username=pass_info['username'],
                         user_agent='whisky_reddit')

    # array of revies added to dataframe
    reviews = []
    n_rows = dataframe.shape[0]
    i = 0
    for index, row in dataframe.iterrows():
        # Pull review link and user
        i += 1
        print(i)
        lnk = row['Link To Reddit Review']
        usr = row["Reviewer's Reddit Username"]

        # Here we pull all top level comments for the review usr
        review_top_level_comments = []

        #  Look through comment tree for top level comments made by reviewer
        # Try statement is used since some submissions maybe from banned subreddits "r/scotchswap"
        try:
            # Identify submission from URL
            submission = reddit.submission(url=lnk)

            # Look through comment tree for top level comments made by reviewer
            for top_level_comment in submission.comments:
                # More comments object represents "load more comments" link in thread
                # Loop past MoreComments to load more more comments
                if isinstance(top_level_comment, MoreComments):
                    continue
                if (top_level_comment.author == usr):
                    review_top_level_comments.append(top_level_comment.body)
        except:
            pass

        n_comments = len(review_top_level_comments)

        # If no comments found pass missing value
        # elif only one comment found then keep that comment
        # else: keep longest length comment
        if not review_top_level_comments:
            reviews.append(np.nan)
        elif n_comments == 1:
            reviews.append(review_top_level_comments[0])
        else:
            comment_lengths = [len(comment)
                               for comment in review_top_level_comments]
            max_length_loc = comment_lengths.index(max(comment_lengths))
            reviews.append(review_top_level_comments[max_length_loc])

    # store values in dataframe
    dataframe['review_text'] = reviews

    return dataframe


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
            self.whisky_archive['Timestamp'], errors='coerce')

        # strip spaces and lowercase characters
        self.whisky_archive['Whisky Region or Style'] = _strip_and_lower_series(
            self.whisky_archive['Whisky Region or Style'])

        # Alternate spellings of whisky styles are replaced and style are prioritized over location.
        replace_dict = {'bourbon/america': 'bourbon',
                        'blended speyside scotch': 'speyside',
                        'lowland / grain': 'lowland',
                        'borubon': 'bourbon',
                        'highlands': 'highland'}

        self.whisky_archive['Whisky Region or Style'] = self.whisky_archive['Whisky Region or Style'].replace(
            replace_dict)

        # Only include styles with at least a 100 reviews
        keep_styles = _keep_frequent_(
            series=self.whisky_archive['Whisky Region or Style'], count=100)

        # Only keep styles in keep_styles array
        self.whisky_archive = self.whisky_archive[self.whisky_archive['Whisky Region or Style'].isin(
            keep_styles)]

        # strip spaces from reviewer names
        self.whisky_archive["Reviewer's Reddit Username"] = self.whisky_archive["Reviewer's Reddit Username"].str.strip()

       # process rating data. strip spaces and lowercase charcters
       # Some users inputed there scores with a divisor. Those are being removed so only the numerator remains
       # convert ratings to float
        self.whisky_archive['Reviewer Rating'] = _strip_and_lower_series(
            self.whisky_archive['Reviewer Rating'])
        self.whisky_archive['Reviewer Rating'] = self.whisky_archive['Reviewer Rating'].str.replace(
            '/100', '')
        self.whisky_archive['Reviewer Rating'] = self.whisky_archive['Reviewer Rating'].astype(
            float, errors='ignore')

        # Validate URLs
        self.whisky_archive = _url_parse_(
            dataframe=self.whisky_archive, url_col='Link To Reddit Review')
        self.whisky_archive = _url_fix_(
            post_url_parse_df=self.whisky_archive, url_col='Link To Reddit Review')

        # URL must point to reddit.com
        self.whisky_archive = self.whisky_archive[self.whisky_archive['Link To Reddit Review'].str.contains(
            'reddit.com')]

    def scrape_reviews(self):

        self.whisky_archive = _scrape_reddit_reviews_(
            pass_loc='pass_info.json', dataframe=self.whisky_archive)

    def get_dataframe(self):
        """
        Returns processed dataframe.
        """
        return self.whisky_archive


if __name__ == "__main__":
    archive = whisky_archive_processor()
    archive.process(
        file_loc='../../data/raw_data/Reddit Whisky Network Review Archive.xlsx')
    archive.scrape_reviews()
    out_df = archive.get_dataframe()
    os.chdir('../../data/interim_data/')
    out_df.to_csv('whisky_archive_scraped_reviews.csv', sep='|')
