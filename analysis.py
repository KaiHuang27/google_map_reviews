import pygsheets
import pymongo
import pandas as pd
import configparser


config = configparser.ConfigParser()
config.read('config.ini')
URL = config['GOOGLESHEET']['URL']

client = pymongo.MongoClient(host='mongodb://db:27017/')
db = client.google_map

gc = pygsheets.authorize(service_account_file='./sheets_key.json')
sheet = gc.open_by_url(URL)

place_data = pd.DataFrame(db.places.find({'name': {'$regex': '王品牛排'}}))
place_data.drop_duplicates(subset='place_id', inplace=True)

review_data = pd.DataFrame(list(db.reviews.find({'nid': {'$in': place_data.nid.tolist()}})))
review_data.drop_duplicates(subset='post_id', inplace=True)
review_data['published_time'] = pd.to_datetime(review_data.published_time.astype(int), unit='ms') # timestamp to datetime
review_data.sort_values('published_time', ascending=False, inplace=True)
review_data = review_data.merge(place_data[['nid', 'name']], how='left', on='nid')

# save places
place_sheet = sheet.worksheet_by_title('places')
place_sheet.clear()
place_sheet.set_dataframe(place_data, start='A1')

# save reviews
review_sheet = sheet.worksheet_by_title('reviews')
review_sheet.clear()
r_data = review_data[review_data.content != 'null']
r_data = r_data[['_id', 'user_name', 'content', 'rating', 'user_id', 'post_id',
                 'published_time', 'nid', 'keywords', 'tags', 'name']]
review_sheet.set_dataframe(r_data, start='A1')

# rating
review_data['month'] = review_data.published_time.dt.month
all_rating = review_data.groupby(['name', 'month'])['rating'].mean().reset_index()
all_rating['tag'] = 'all'
tags = ['taste', 'service', 'environment', 'price']
for t in tags:
    temp_rating = review_data[review_data.tags.astype(str).str.contains(t)].groupby(['name', 'month'])['rating'].mean().reset_index()
    temp_rating['tag'] = t
    all_rating = all_rating.append(temp_rating)

all_rating.reset_index(drop=True, inplace=True)

rating_sheet = sheet.worksheet_by_title('rating')
rating_sheet.clear()
rating_sheet.set_dataframe(all_rating, start='A1')

# keywords
kw_result = []
groups = review_data.groupby('name')
for name, g in groups:
    keywords = g.keywords.apply(pd.Series).stack().reset_index(drop=True)
    keywords = keywords.value_counts().head(15).index.tolist()
    kw_result.append({'name': name, 'keywords': keywords, 'rating': round(g.rating.mean(), 2)})
kw_result = pd.DataFrame(kw_result)

kw_sheet = sheet.worksheet_by_title('keywords')
kw_sheet.clear()
kw_sheet.set_dataframe(kw_result, start='A1')