# pip install chart_studio
# pip install datapane
%matplotlib inline
import pandas 
from sklearn.model_selection import train_test_split
import numpy as np
import time
import joblib
import Recommenders as Recommenders
import Evaluation as Evaluation

#Read useris, songid, listen
triplets_file = '10000.txt'
song_metadata_file = 'song_data.csv'
#Create dataframe for triplet file
song_df1 = pd.read_table(triplets_file, header = None)
song_df1.columns = ['user_id', 'song_id', 'listen_count']
#Dataframe for metadata file
song_df2 = pd.read_csv(song_metadata_file)

#Merge 2 data file
song_df = pd.merge(song_df1, song_df2.drop_duplicates(['song_id']),on="song_id",how='left')

song_df.head()
#Check length of data
print("Total number of elements:", len(song_df))
#Create subset of data
song_df = song_df.head(10000)

#Merge song title and song artist into one column
song_df['song'] = song_df['title'].map(str)+"-"+song_df['artist_name']
#Number of unique users
users = song_df['user_id'].unique()
len(users)
#Number of unique songs:
songs = song_df['song'].unique()
len(songs)

#Split data to train data and test data
train_data , test_data = train_test_split(song_df,test_size = 0.20, random_state = 0)
print(train_data.head(5))

# Simple popularity based recommender
pm = Recommenders.popularity_recommender()
pm.create(train_data,'user_id','song')
# Using recommender to make some recommendation
user_id = users[35]
pm.recommend(user_id)
is_model = Recommenders.item_similarity_recommender()
is_model.create(train_data,'user_id','song')
user_item = is_model.get_user_items(user_id)
print("Users history music data:\n",user_item)
print("Recommendations for User id",user_id,":")
is_model.recommend(user_id)