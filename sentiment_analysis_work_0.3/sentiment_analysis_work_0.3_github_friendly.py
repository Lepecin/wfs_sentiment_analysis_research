# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
# Essential packages

# Import praw, the Python Reddit API Wrapper
import praw

# Import datetime, for simple date and time storing
import datetime

# Import pandas, for data storage
import pandas as pd

# Import matplotlib.pyplot, for creating plots
import matplotlib.pyplot as plt

# Import numpy, for array manipulation
import numpy as np

# Extra packages

# Import pprint
# import pprint

# Import seaborn
# import seaborn as sns

# Import psaw
# import psaw

# Import pandas data reader
# import pandas_datareader as web

# Import dateutil
# from dateutil import relativedelta


# %%
# NLP packages

# Import nltk
import nltk

# Download the Vader lexicon dataset for
# sentiment analysis
# nltk.download('vader_lexicon')

# Import the sentiment intensity analyser
# function from nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

# Create the SIA as a simple referable
# object
sia = SIA()

# Reference code for extracting sentiment
# data

#results = []
#line = "APLL is a nice stock"
#score = sia.polarity_scores(line)
#results.append(score)
#pd.DataFrame.from_records(results)


# %%
# For installing psaw package
# Done using installed pip package installer
# pip install psaw

# Set up a Read-Only Reddit client
reddit = praw.Reddit(
    
    # Input the client id
    client_id = "<client_id>",
    
    # Input the client secret
    client_secret = "<client_secret>",
    
    # Set the name of the client
    user_agent = "<user_agent>"
)


# %%
# Import the re package
import re

# Import the collections package
import collections

# ---------------------------------------------------

# Function for creating dictionary, counting mentions
# of any stock tickers it can find

# ---------------------------------------------------


def remove_redundant(txt):

  # re function for striping all redundant characters 
  # in input txt
  txt = re.sub('[^^/A-Z]', ' ', txt)

  # Split the stripped text into chunks
  txt = txt.split()

  # Use Counter to find frequency of elements
  frequency = collections.Counter(list(txt))

  # Turn the frequency object into a dictionary
  frequency = dict(frequency)

  # Filter any entries in frequency that do not belong
  # to the list of stock ticker symbols
  filtered_frequency = {k: v for k, v in frequency.items() if k in st_tickers}

  # Return the modified txt
  return filtered_frequency


# ---------------------------------------------------

# Function for adding entries for two
# dictionaries

# ---------------------------------------------------


def dict_combine(dict_1, dict_2):

    # Define the counter version of first dictionary
    a_counter = collections.Counter(dict_1)

    # Do same for next dictionary
    b_counter = collections.Counter(dict_2)

    # Define the output dictionary as the sum of
    # counters of input dictionaries
    dict_3 = dict(a_counter + b_counter)

    # Output the combined dictionaries
    return dict_3


# ---------------------------------------------------

# Data extraction function :)
# The holy grail
# Add text and id, then await magic to retrieve
# your data :D

# ---------------------------------------------------


def data_extract(comm, txt):
    
    # Convert the comment number into a dataset
    comm_unitdata = pd.DataFrame(data = [comm], columns = ['comment'])

    # Retrieve the ticker counts data from input
    # text
    txt_data = remove_redundant(txt).items()

    # Convert the ticker counts data into
    # dataframe object
    txt_data = pd.DataFrame(data = txt_data, columns = ['ticker','count'])

    # Concatenate the comm and txt_data datasets
    comm_data = pd.concat([comm_unitdata ,txt_data], sort = False, axis = 1)

    # Modify the comment column to be filled with
    # the comment number
    comm_data['comment'] = comm_data['comment'].fillna(comm)

    # Set comm_data to be a none object is it
    # has missing data
    if comm_data.isnull().any().any():
        comm_data = None

    # Extract the sentiment intensity data from
    # the text
    score = sia.polarity_scores(txt)

    # Record the sentiment intensity data into
    # a data frame
    sent_data = pd.DataFrame.from_records([score])

    # Concatenate the sentiment dataset with the
    # comm_unitdata dataset
    sent_data = pd.concat([comm_unitdata ,sent_data], sort = False, axis = 1)

    # Return the two datasets
    return sent_data, comm_data


# ---------------------------------------------------

# Function for creating datasets for
# ticker counts, comment dates, and
# sentiment data

# ---------------------------------------------------


def ticker_puller(submission):

    # Create three empty datasets

    # One for setniment intensity data
    sentiment_intensity_data = pd.DataFrame(data = [])

    # One for ticker counts data
    ticker_counts_data = pd.DataFrame(data = [])

    # Last for the dates and text data
    comment_datetxt_data = pd.DataFrame(data = [])


    # Start extracting data from the title of
    # the submission, which will be comment zero

    # Extract the title text
    title_txt = submission.title

    # Extract the ticker and sentiment data from
    # the title text
    title_vanilla_extract = data_extract(0, title_txt)


    # Now extract data about the time of
    # the submission, which will be the time
    # for comment zero (the title)

    # Extract the submission time
    title_time_data = submission.created_utc

    # Convert the extracted time from unix
    # timetstamp form into datetime form
    title_time_data = datetime.datetime.fromtimestamp(int(title_time_data))

    # Place the comment number (zero), the extracted
    # time and the submission title into a single
    # data frame
    title_time_data = pd.DataFrame(data = {'comment': [0], 'time': [title_time_data], 'text': [submission.title]})

    # Finally, concatenate the date/text data
    # for the title into the main date and text
    # dataset
    comment_datetxt_data = pd.concat([comment_datetxt_data, title_time_data], sort = False)


    # Here, extract the sentiment data
    # produced by the data extract function
    # on title text
    title_vanilla_sentiment = title_vanilla_extract[0]

    # Append the title sentiment data to
    # the main sentiment data set
    sentiment_intensity_data = pd.concat([sentiment_intensity_data, title_vanilla_sentiment], sort = False)


    # Extract the ticker counts data
    # produced by the data extract function
    title_vanilla_ticker_counts = title_vanilla_extract[1]

    # Append the data to the main ticker counts
    # dataset
    ticker_counts_data = pd.concat([ticker_counts_data, title_vanilla_ticker_counts], sort = False)


    # Here, we move on to extracting data
    # from the comments and replies for the
    # submission

    # Create the list of comments
    # for the submission
    comments_list = submission.comments

    # Remove redundant replies elements
    # from the comments list
    comments_list.replace_more(limit = 0)

    # Create the comments queue, excluding the
    # first (zeroth) comment since it's just
    # data about the submission
    comments_queue = comments_list[1:]

    # Set the counter that will assign
    # comments their numbers
    j_counter = 0


    # Loop through the comments queue until
    # it is empty
    while comments_queue:
        
        # First, extract some data

        # Pop out a comment from the comment queue
        # and set it to subject_comment, the comment
        # analysed this iteration
        subject_comment = comments_queue.pop(0)

        # Set the comment body as the text
        # to be analysed
        txt = subject_comment.body

        # Extract ticker and sentiment data from
        # the comment
        vanilla_extract = data_extract(j_counter+1, txt)

        # Here we create a datetxt entry

        # Extract the time of the comment
        time_data = subject_comment.created_utc

        # Change the time from unix timestamp
        # to datetime format
        time_data = datetime.datetime.fromtimestamp(int(time_data))

        # Record the time, the comment body and the comment
        # value into a dataframe
        time_data = pd.DataFrame(data = {'comment': [j_counter+1], 'time': [time_data], 'text': [txt]})

        # Append the time_data data frame to the
        # main datetxt data frame
        comment_datetxt_data = pd.concat([comment_datetxt_data, time_data], sort = False)

        # Extract and recorde sentiment data

        vanilla_sentiment = vanilla_extract[0]

        sentiment_intensity_data = pd.concat([sentiment_intensity_data, vanilla_sentiment], sort = False)

        # Extract and record ticker count data

        vanilla_ticker_counts = vanilla_extract[1]

        ticker_counts_data = pd.concat([ticker_counts_data, vanilla_ticker_counts], sort = False)


        # Advance the counter by one
        j_counter += 1

        # Append the replies of the comment to the
        # comment queue
        comments_queue.extend(subject_comment.replies)


    # Following code resets the index for each of
    # three main datasets and removes the created
    # index column
    sentiment_intensity_data = sentiment_intensity_data.reset_index().drop(labels = 'index', axis = 1)

    ticker_counts_data = ticker_counts_data.reset_index().drop(labels = 'index', axis = 1)

    comment_datetxt_data = comment_datetxt_data.reset_index().drop(labels = 'index', axis = 1)



    return sentiment_intensity_data, ticker_counts_data, comment_datetxt_data


# ---------------------------------------------------

# Function for creating datasets for
# ticker counts, comment dates, and
# sentiment data (for submission
# selftext)

# ---------------------------------------------------


def submission_ticker_puller(comm, submission):



    sentiment_intensity_data = pd.DataFrame(data = [])

    ticker_counts_data = pd.DataFrame(data = [])

    comment_datetxt_data = pd.DataFrame(data = [])



    self_txt = submission.selftext

    self_vanilla_extract = data_extract(comm, self_txt)



    self_time_data = submission.created_utc

    self_time_data = datetime.datetime.fromtimestamp(int(self_time_data))

    self_time_data = pd.DataFrame(data = {'submission': [comm], 'time': [self_time_data], 'text': [self_txt]})

    comment_datetxt_data = pd.concat([comment_datetxt_data, self_time_data], sort = False)



    self_vanilla_sentiment = self_vanilla_extract[0]

    sentiment_intensity_data = pd.concat([sentiment_intensity_data, self_vanilla_sentiment], sort = False)



    self_vanilla_ticker_counts = self_vanilla_extract[1]

    ticker_counts_data = pd.concat([ticker_counts_data, self_vanilla_ticker_counts], sort = False)



    sentiment_intensity_data = sentiment_intensity_data.reset_index().drop(labels = 'index', axis = 1)

    ticker_counts_data = ticker_counts_data.reset_index().drop(labels = 'index', axis = 1)

    comment_datetxt_data = comment_datetxt_data.reset_index().drop(labels = 'index', axis = 1)



    return sentiment_intensity_data, ticker_counts_data, comment_datetxt_data


# ---------------------------------------------------


# %%
# Upload the stock ticker symbols from the NASDAQ Symbols.csv
ticker_NASDAQ = pd.read_csv('Data Sets (Ticker Symbols)/NASDAQ Symbols.csv')

# Upload the stock ticker symbols from the NYSE Symbols.csv
ticker_NYSE = pd.read_csv('Data Sets (Ticker Symbols)/NYSE Symbols.csv')

# Upload the stock ticker symbols from the AMEX Symbols.csv
ticker_AMEX = pd.read_csv('Data Sets (Ticker Symbols)/AMEX Symbols.csv')

# Create the to be set of unique stock ticker symbols
st_tickers = set()

# Update the st_tickers set with ticker codes from
# NASDAQ, NYSE and AMEX
st_tickers.update(set(ticker_NASDAQ['Symbol']))
st_tickers.update(set(ticker_NYSE['Symbol']))
st_tickers.update(set(ticker_AMEX['Symbol']))

# Code for determining if the ticker code is in
# the st_tickers set
# 'A' in st_tickers


# %%
# Code for excluding single letter ticker symbols

# Create list of ticker symbols to exclude
exclusion_list = [i for i in st_tickers if len(i) == 1]

# Modify st_tickers to exclude the unneeded symbols
st_tickers.difference_update(exclusion_list)


# %%
# Access a subreddit

# Examples of stock market related subreddits

# 'wallstreetbets'
# 'stocks'
# 'investing'
# 'pennystocks'
# 'RobinHood'

# Set the subreddit to analyse
subreddit = reddit.subreddit('wallstreetbets')

# Print the subreddit's name
print(subreddit.display_name)

# Print the subreddit's title
# print(subreddit.title)

# Print the subreddit's description
# print(subreddit.description)


# %%
# Create list object into which submissions are placed
submission_list = []

# Cycle through submissions in hot category
# limit parameter states number of submissions to record
# In case of none, there is no limit to how many to record
# So all are recorded
for submission in subreddit.hot(limit = None):
    
    submission_list.append(submission)


# %%
len(submission_list)


# %%
# Choose submission to analyse
i = 0

# Set the submission
submission = submission_list[i]

# Details about submission
print(submission.title)
print()
print(submission.selftext)
print()
print(submission.author)


# %%
# Code for printing the body text of
# all comments and replies of a submission

# Create a list of comments
comments_list = submission.comments

# Separates replies from list of comments
comments_list.replace_more(limit = 0)

# List of comments without replies
# print(comments_list.list())

# Transform the comment list into an
# actual list object
comments_queue = comments_list[1:]

# Cycle through all comments in list
while comments_queue:
    
    # Set the printed comment as the one popped
    # from the comment queue, so it's read and
    # removed from the queue
    comment = comments_queue.pop(0)
    
    # Print the comment's body
    print(comment.body)
    print()
    
    # Other information about the comment is
    # readily avaliable in this window here
    
    # Extend the comment queue with the replies
    # of the comment printed, then move on to the
    # next comment
    comments_queue.extend(comment.replies)


# %%
final_sentiment_dataset = pd.DataFrame(data = [])

final_ticker_dataset = pd.DataFrame(data = [])

final_datetxt_dataset = pd.DataFrame(data = [])



submission_queue = submission_list

subm = 0



while submission_queue:

    submission = submission_queue.pop(0)

    subm_unitdata = pd.DataFrame(data = [subm], columns = ['submission'])



    final_data = ticker_puller(submission)



    subm_data_0 = pd.concat([subm_unitdata, final_data[0]], sort = False, axis = 1)

    subm_data_0['submission'] = subm_data_0['submission'].fillna(subm)



    subm_data_1 = pd.concat([subm_unitdata, final_data[1]], sort = False, axis = 1)

    subm_data_1['submission'] = subm_data_1['submission'].fillna(subm)



    subm_data_2 = pd.concat([subm_unitdata, final_data[2]], sort = False, axis = 1)

    subm_data_2['submission'] = subm_data_2['submission'].fillna(subm)



    final_sentiment_dataset = pd.concat([final_sentiment_dataset, subm_data_0], sort = False)

    final_ticker_dataset = pd.concat([final_ticker_dataset, subm_data_1], sort = False)

    final_datetxt_dataset = pd.concat([final_datetxt_dataset, subm_data_2], sort = False)



    final_sentiment_dataset = final_sentiment_dataset.reset_index().drop(labels = 'index', axis = 1)

    final_ticker_dataset = final_ticker_dataset.reset_index().drop(labels = 'index', axis = 1)

    final_datetxt_dataset = final_datetxt_dataset.reset_index().drop(labels = 'index', axis = 1)



    print('Submission', str(subm), 'is added')

    subm += 1


# %%
# final_sentiment_dataset.to_csv('data_sets_usable/final_sentiment_dataset.csv')

# final_ticker_dataset.to_csv('data_sets_usable/final_ticker_dataset.csv')

# final_datetxt_dataset.to_csv('data_sets_usable/final_datetxt_dataset.csv')


