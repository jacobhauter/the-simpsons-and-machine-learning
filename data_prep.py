import numpy as np
import pandas as pd
pd.options.display.float_format = '{:20,.3f}'.format
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import seaborn as sns
sns.set()
from statsmodels.stats.outliers_influence import variance_inflation_factor

episodes = pd.read_csv('simpsons_episodes.csv')
characters = pd.read_csv('simpsons_characters.csv')
locations = pd.read_csv('simpsons_locations.csv')
script_lines = pd.read_csv('simpsons_script_lines.csv')

# Data Preparation

episodes_cleaned = episodes[[
    'id',
    'title',
    'imdb_rating',
    'us_viewers_in_millions',
    'season',
    'number_in_season',
    'number_in_series',
    'original_air_year',
    'original_air_date'
]]

episodes_cleaned.rename(columns={'id':'episode_id'}, inplace=True)

# Find words per location, words per character, and words per episode

script_lines['word_count'] = pd.to_numeric(script_lines['word_count'].fillna(0), errors='coerce')
script_lines['word_count'] = script_lines['word_count'].fillna(0).apply(np.int64)

script_lines_words_per_location = script_lines[['episode_id','location_id','word_count']].groupby(['episode_id','location_id']).sum().reset_index()

script_lines_words_per_character = script_lines[['episode_id','character_id','word_count']].groupby(['episode_id','character_id']).sum().reset_index()

script_lines_words_per_episode = script_lines[['episode_id','word_count']].groupby(['episode_id',]).sum().reset_index()

# Location word count ratio

script_lines_words_per_location_and_episode = script_lines_words_per_location.merge(script_lines_words_per_episode, how='inner', on='episode_id')
script_lines_words_per_location_and_episode['location_word_count_ratio'] = script_lines_words_per_location_and_episode['word_count_x'] / script_lines_words_per_location_and_episode['word_count_y']

# Character word count ratio

script_lines_words_per_character_and_episode = script_lines_words_per_character.merge(script_lines_words_per_episode, how='inner', on='episode_id')
script_lines_words_per_character_and_episode['character_word_count_ratio'] = script_lines_words_per_character_and_episode['word_count_x'] / script_lines_words_per_character_and_episode['word_count_y']

# Character words per episode

character_words_per_episode = script_lines_words_per_character_and_episode.pivot(index='episode_id', columns='character_id', values='word_count_x').fillna(0).astype(np.int64)
character_words_per_episode.columns = ['words_per_character_' + str(col) for col in character_words_per_episode.columns]

# Location word count ratio

script_lines_words_per_location_and_episode = script_lines_words_per_location.merge(script_lines_words_per_episode, how='inner', on='episode_id')
script_lines_words_per_location_and_episode['location_word_count_ratio'] = script_lines_words_per_location_and_episode['word_count_x'] / script_lines_words_per_location_and_episode['word_count_y']

# Location words per episode

location_words_per_episode = script_lines_words_per_location_and_episode.pivot(index='episode_id', columns='location_id', values='word_count_x').fillna(0).astype(np.int64)
location_words_per_episode.columns = ['words_per_location_' + str(col) for col in location_words_per_episode.columns]

# Merge character words per episode to main
episodes_cleaned_with_character_words_per_episode = episodes_cleaned.merge(character_words_per_episode, how='left', on=['episode_id'])

# Merge location words per episode to main
episodes_cleaned_with_location_words_per_episode = episodes_cleaned.merge(location_words_per_episode, how='left', on=['episode_id'])

# Merge both dataframes
episodes_cleaned_with_character_and_location_words_per_episode = episodes_cleaned_with_character_words_per_episode.merge(location_words_per_episode.copy(), how='left', on=['episode_id'])

df = episodes_cleaned_with_character_and_location_words_per_episode

cols = df.columns.drop(['title','imdb_rating','us_viewers_in_millions','original_air_date'])

df[cols] = df[cols].fillna(0).astype(np.int64)

#df.to_csv('Cdf_export.csv', index=False)

episodes_cleaned_with_character_and_location_words_per_episode
