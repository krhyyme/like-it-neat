import pandas as pd
import os

"""
This script uses pandas to convert the reddit Whiskey review data set into a csv. Run from the src directory
"""

os.chdir("../../data/raw_data")

df = pd.read_excel("Reddit Whisky Network Review Archive.xlsx", dtype=str)

df.to_csv("reddit_whisky_network_review_archive.csv", index=False)
