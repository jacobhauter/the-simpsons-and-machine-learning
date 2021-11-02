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

df_export = pd.read_csv('df_export.csv')

test_data = df_export.drop([
    'episode_id',
    'title',
    'us_viewers_in_millions',  
    'season',  
    'number_in_season',  
    'number_in_series', 
    'original_air_year', 
    'original_air_date'
],axis=1)

test_data = test_data.dropna()

# Visualization: Ratings through Time
epsiode_ratings_time = episodes[['season','imdb_rating']]
epsiode_ratings_time = epsiode_ratings_time.groupby(['season'], as_index=False).mean().round(2)
epsiode_ratings_time
Year = epsiode_ratings_time['season']
Rating = epsiode_ratings_time['imdb_rating']
plt.plot(Year, Rating, color="black")
plt.title('Average Rating by Season')
plt.xlabel('Season')
plt.ylabel('IMDB Rating')
plt.show()

# Visualization: Viewership through Time
epsiode_ratings_time = episodes[['season','us_viewers_in_millions']]
epsiode_ratings_time = epsiode_ratings_time.groupby(['season'], as_index=False).mean().round(2)
epsiode_ratings_time
Year = epsiode_ratings_time['season']
Rating = epsiode_ratings_time['us_viewers_in_millions']
plt.plot(Year, Rating, color="black")
plt.title('Average Viewership by Season')
plt.xlabel('Season')
plt.ylabel('Viewers in Millions')
plt.show()

# Visualization: Top Rated Epsiodes
episode_ratings = episodes[['imdb_rating','title']]
episode_ratings_top = episode_ratings.sort_values('imdb_rating',ascending=False).head(15)
plt.figure(figsize=(10,6))
episode_ratings_top = episode_ratings_top.sort_values('imdb_rating',ascending=True)
x = episode_ratings_top['imdb_rating']
y = episode_ratings_top['title']
y_pos = [i for i, _ in enumerate(y)]
plt.barh(y_pos, x, color='yellow')
plt.ylabel("Epsiode")
plt.xlabel("IMDB Rating")
plt.title("Top Rated Epsiodes")
plt.yticks(y_pos, y)
plt.xlim([0, 10])
plt.show()

# Visualization: Lowest Rated Epsiodes
episode_ratings = episodes[['imdb_rating','title']]
episode_ratings_low = episode_ratings.sort_values('imdb_rating',ascending=True).head(15)
plt.figure(figsize=(10,6))
episode_ratings_low = episode_ratings_low.sort_values('imdb_rating',ascending=False)
x = episode_ratings_low['imdb_rating']
y = episode_ratings_low['title']
y_pos = [i for i, _ in enumerate(y)]
plt.barh(y_pos, x, color='yellow')
plt.ylabel("Epsiode")
plt.xlabel("IMDB Rating")
plt.title("Lowest Rated Epsiodes")
plt.yticks(y_pos, y)
plt.xlim([0, 10])
plt.show()

# Get words total
test_data['words_total'] = test_data.drop('imdb_rating', axis=1).sum(axis=1)
sns.distplot(test_data['words_total'], color="y")
test_data_cleaned = test_data[test_data['words_total'] < 7000]
test_data_cleaned = test_data_cleaned[test_data_cleaned['words_total'] > 0]
sns.distplot(test_data_cleaned['words_total'], color="y")

# Find top characters
character_word_sums = [col for col in test_data_cleaned if col.startswith('words_per_character_')]
test_data_cleaned_character_words = test_data_cleaned[character_word_sums]
test_data_cleaned_character_words = test_data_cleaned_character_words.T
test_data_cleaned_character_words.reset_index(level=0, inplace=True)
test_data_cleaned_character_words.rename(columns={'index': 'Character'}, inplace=True)
test_data_cleaned_character_words['character_words_total'] = test_data_cleaned_character_words.sum(axis=1)
test_data_cleaned_character_words_percentile = test_data_cleaned_character_words[['Character','character_words_total']]
test_data_cleaned_character_words_percentile = test_data_cleaned_character_words_percentile.sort_values('character_words_total',ascending=False)
test_data_cleaned_character_words_percentile['Percentile'] = test_data_cleaned_character_words_percentile['character_words_total'] / test_data_cleaned_character_words_percentile['character_words_total'].sum()
test_data_cleaned_character_words_percentile['Running_Percentile'] = test_data_cleaned_character_words_percentile['Percentile'].cumsum()
specified_percentile = 0.8
test_data_cleaned_character_words_percentile_specified = test_data_cleaned_character_words_percentile[test_data_cleaned_character_words_percentile['Running_Percentile'] < specified_percentile]
test_data_cleaned_character_words_top_characters = test_data_cleaned_character_words_percentile_specified['Character'].to_frame().merge(test_data_cleaned_character_words,how='left',on='Character')
test_data_cleaned_character_words_top_characters

# Visualization: Average Character Words Per Episode 
test_data_cleaned_character_words_top_characters_plot = test_data_cleaned_character_words_top_characters[['Character','character_words_total']].sort_values('character_words_total',ascending=False).head(25)
test_data_cleaned_character_words_top_characters_plot['id'] = test_data_cleaned_character_words_top_characters_plot['Character'].str.lstrip('words_per_character_').astype(float).astype(int)
test_data_cleaned_character_words_top_characters_plot = pd.merge(test_data_cleaned_character_words_top_characters_plot,characters,on='id',how='left')
test_data_cleaned_character_words_top_characters_plot['average_words_per_episode'] = (test_data_cleaned_character_words_top_characters_plot['character_words_total'] / test_data_cleaned_character_words_top_characters.count(axis='columns').max()).round().astype(int)
plt.figure(figsize=(10,6))
test_data_cleaned_character_words_top_characters_plot = test_data_cleaned_character_words_top_characters_plot.sort_values('average_words_per_episode',ascending=True)
x = test_data_cleaned_character_words_top_characters_plot['average_words_per_episode']
y = test_data_cleaned_character_words_top_characters_plot['name']
y_pos = [i for i, _ in enumerate(y)]
plt.barh(y_pos, x, color='yellow')
plt.ylabel("Character")
plt.xlabel("Words")
plt.title("Average Character Words Per Episode")
plt.yticks(y_pos, y)
plt.show()

# Find top locations
location_word_sums = [col for col in test_data_cleaned if col.startswith('words_per_location_')]
test_data_cleaned_location_words = test_data_cleaned[location_word_sums]
test_data_cleaned_location_words = test_data_cleaned_location_words.T
test_data_cleaned_location_words.reset_index(level=0, inplace=True)
test_data_cleaned_location_words.rename(columns={'index': 'Location'}, inplace=True)
test_data_cleaned_location_words['location_words_total'] = test_data_cleaned_location_words.sum(axis=1)
test_data_cleaned_location_words_percentile = test_data_cleaned_location_words[['Location','location_words_total']]
test_data_cleaned_location_words_percentile = test_data_cleaned_location_words_percentile.sort_values('location_words_total',ascending=False)
test_data_cleaned_location_words_percentile['Percentile'] = test_data_cleaned_location_words_percentile['location_words_total'] / test_data_cleaned_location_words_percentile['location_words_total'].sum()
test_data_cleaned_location_words_percentile['Running_Percentile'] = test_data_cleaned_location_words_percentile['Percentile'].cumsum()
specified_percentile = 0.67
test_data_cleaned_location_words_percentile_specified = test_data_cleaned_location_words_percentile[test_data_cleaned_location_words_percentile['Running_Percentile'] < specified_percentile]
test_data_cleaned_location_words_top_locations = test_data_cleaned_location_words_percentile_specified['Location'].to_frame().merge(test_data_cleaned_location_words,how='left',on='Location')
test_data_cleaned_location_words_top_locations

# Visualization: Average Location Words Per Episode 
test_data_cleaned_location_words_top_locations_plot = test_data_cleaned_location_words_top_locations[['Location','location_words_total']].sort_values('location_words_total',ascending=False).head(25)
test_data_cleaned_location_words_top_locations_plot['id'] = test_data_cleaned_location_words_top_locations_plot['Location'].str.lstrip('words_per_location_').astype(float).astype(int)
test_data_cleaned_location_words_top_locations_plot = pd.merge(test_data_cleaned_location_words_top_locations_plot,locations,on='id',how='left')
test_data_cleaned_location_words_top_locations_plot['average_words_per_episode'] = (test_data_cleaned_location_words_top_locations_plot['location_words_total'] / test_data_cleaned_location_words_top_locations.count(axis='columns').max()).round().astype(int)
plt.figure(figsize=(10,6))
test_data_cleaned_location_words_top_locations_plot = test_data_cleaned_location_words_top_locations_plot.sort_values('average_words_per_episode',ascending=True)
x = test_data_cleaned_location_words_top_locations_plot['average_words_per_episode']
y = test_data_cleaned_location_words_top_locations_plot['name']
y_pos = [i for i, _ in enumerate(y)]
plt.barh(y_pos, x, color='yellow')
plt.ylabel("Location")
plt.xlabel("Words")
plt.title("Average Location Words Per Episode")
plt.yticks(y_pos, y)
plt.show()

# Re-Transpose Top Characters
test_data_cleaned_character_words_top_character_clean = test_data_cleaned_character_words_top_characters
test_data_cleaned_character_words_top_character_clean = test_data_cleaned_character_words_top_character_clean.set_index('Character').T
test_data_cleaned_character_words_top_character_clean = test_data_cleaned_character_words_top_character_clean[:-1]
test_data_cleaned_character_words_top_character_clean

# Remove Locations with outlier words
test_data_cleaned_location_words_top_locations_clean = test_data_cleaned_location_words_top_locations[test_data_cleaned_location_words_top_locations['location_words_total'] < 100000]

# Re-Transpose Top Locations
test_data_cleaned_location_words_top_locations_clean = test_data_cleaned_location_words_top_locations_clean.set_index('Location').T
test_data_cleaned_location_words_top_locations_clean = test_data_cleaned_location_words_top_locations_clean[:-1]
test_data_cleaned_location_words_top_locations_clean

prepared_dfs = [test_data_cleaned['imdb_rating'],test_data_cleaned_character_words_top_character_clean,test_data_cleaned_location_words_top_locations_clean]

test_data_cleaned_prepared = pd.concat(prepared_dfs, axis=1)

sns.distplot(test_data_cleaned_prepared['imdb_rating'], color="y")

# Calculate Variance Inflation Factors to find Multicollinear Variables
variables = test_data_cleaned_prepared.drop(['imdb_rating'],axis=1)
vif = pd.DataFrame()
vif['VIF'] = [variance_inflation_factor(variables.values, i) for i in range(variables.shape[1])]
vif['VIF'] = vif['VIF'].round(2)
vif['Feature'] = variables.columns
vif = vif[['Feature','VIF']]
vif = vif.sort_values('VIF',ascending=False)
vif_multicollinearity = vif[vif['VIF'] > 5]
vif_multicollinearity[['Feature','VIF']]
vif_multicollinearity_list = vif_multicollinearity['Feature'].tolist()

# Set targets and inputs
targets = test_data_cleaned_prepared['imdb_rating']
inputs = test_data_cleaned_prepared.drop(['imdb_rating'],axis=1)
#inputs = inputs.drop(columns=vif_multicollinearity_list)
#inputs = inputs.drop(['words_per_character_1.0','words_per_character_2.0','words_per_character_8.0','words_per_character_9.0'],axis=1)

# Standardize inputs
scaler = StandardScaler()
scaler.fit(inputs)
inputs_scaled = scaler.transform(inputs)

# Set the training and testing and linear regression model
x_train, x_test, y_train, y_test = train_test_split(inputs_scaled, targets, test_size=0.2)
reg = LinearRegression()
reg.fit(x_train,y_train)
y_hat = reg.predict(x_train)
plt.scatter(y_train, y_hat, alpha=0.25)
plt.xlabel('Target IMDB rating', size=15)
plt.ylabel('Predicted IMDB rating',  size=15)
z = np.polyfit(y_train, y_hat, 1)
p = np.poly1d(z)
plt.plot(y_train,p(y_train),"y--",lw=0.5)
plt.show()

# Plot residual distributions.
sns.distplot(y_train - y_hat, color="y")
plt.title('Residuals Distribution')

print('The R-squared value is ' + str(round(reg.score(x_train, y_train),3)) + ', meaning this model explains ' + str(round(reg.score(x_train, y_train),3)*100) + '% of the variability in the data.')

# Regression Summary
reg_summary = pd.DataFrame(inputs.columns.values,columns=['Features'])
reg_summary['Weights'] = reg.coef_
reg_summary_highest_abs = reg_summary.iloc[(-np.abs(reg_summary['Weights'].values)).argsort()].reset_index(drop=True)
reg_summary_highest_abs

# Find most influential characters and locations
most_influential_characters = reg_summary[reg_summary['Features'].str.startswith('words_per_character_')]
most_influential_characters['id'] = most_influential_characters['Features'].str.lstrip('words_per_character_').astype(float).astype(int)
most_influential_characters = pd.merge(most_influential_characters,characters,on='id',how='left')
most_influential_characters = most_influential_characters[['name','Weights']]
most_influential_characters['Weights'] = most_influential_characters['Weights'].round(4)
most_influential_locations = reg_summary[reg_summary['Features'].str.startswith('words_per_location_')]
most_influential_locations['id'] = most_influential_locations['Features'].str.lstrip('words_per_location_').astype(float).astype(int)
most_influential_locations = pd.merge(most_influential_locations,locations,on='id',how='left')
most_influential_locations = most_influential_locations[['name','Weights']]
most_influential_locations['Weights'] = most_influential_locations['Weights'].round(4)

# Test the Model
y_hat_test = reg.predict(x_test)
plt.scatter(y_test, y_hat_test, alpha=0.25)
plt.xlabel('Target imdb_rating', size=15)
plt.ylabel('Predicted imdb_rating',  size=15)
plt.show()

# Find differences between predictions and targets
df_pf = pd.DataFrame(y_hat_test, columns=['Prediction'])
y_test = y_test.reset_index(drop=True)
df_pf['Target'] = y_test
df_pf['Residual'] = df_pf['Target'] - df_pf['Prediction']
df_pf['Difference %'] = np.absolute(df_pf['Residual']/df_pf['Target']*100)
df_pf = df_pf.sort_values(by=['Difference %'])
df_pf
