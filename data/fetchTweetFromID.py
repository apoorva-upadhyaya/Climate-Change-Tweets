import tweepy
import json
import os
import ast
import pandas as pd
import csv
import time

CONSUMER_KEY=""
CONSUMER_SECRET=""
OAUTH_TOKEN=""
OAUTH_TOKEN_SECRET=""

list_ids=[]
df_stu=pd.read_csv("final_data_share.csv.csv", delimiter=";") 
print("df_stu :: ",len(df_stu))

list_ids=df_stu["tweetid"].tolist()

print("len of tweets",len(list_ids),list_ids[:10])


auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
auth.set_access_token(OAUTH_TOKEN, OAUTH_TOKEN_SECRET)
api = tweepy.API(auth,wait_on_rate_limit=True,wait_on_rate_limit_notify=True)

count=0
for single_id in list_ids:
	#print(type(single_id))
	
	try:		
		tweet=api.get_status(single_id, tweet_mode='extended')
		count=count+1
		print(count)
		#<convert <class 'tweepy.models.Status'> to string representation of dict>
		json_str = json.dumps(tweet._json)
		#<convert string representation of dict to dict>
		tweet=json.loads(json_str)
		with open("tweets_data.json","a") as f:
			json.dump(tweet,f)
			f.write("\n")
	except Exception as e:
		print("e ::",e)
		continue
