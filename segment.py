import jieba
import jieba.analyse
import pandas as pd
import pymongo
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


client = pymongo.MongoClient(host='mongodb://localhost:27017/')
db = client.google_map
reviews = db.reviews

jieba.load_userdict("userdict.txt")


data = pd.DataFrame(list(reviews.find()))
data.drop_duplicates(subset='post_id', inplace=True, ignore_index=True)
data = data[~data.content.str.contains('由 Google 提供翻譯') & ~data.content.str.contains('null')].reset_index(drop=True)
data['content'] = data.content.str.replace('[\n，。]', ' ', regex=True)


data['segment'] = data['content'].apply(lambda x: jieba.lcut(x))
for i in range(50, 100):
    print(data.loc[i, 'segment'])

'''
s = SnowNLP(data.segment)
review_tfidf = []
for review_tf in s.tf:
    tfidf = {}
    for word in review_tf:
        tfidf[word] = review_tf[word] * s.idf[word]
    tfidf = dict(sorted(tfidf.items(), key=lambda x: x[1]))
    review_tfidf.append(tfidf)
'''

# bag of words
count_vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w{2,10}\b')
bag_of_words = count_vectorizer.fit_transform(data.segment.str.join(' '))
#pd.DataFrame(bag_of_words.toarray(),columns=count_vectorizer.get_feature_names())
#r.loc[0][r.loc[0] > 0]

# tfidf
transformer = TfidfTransformer()
tfidf = transformer.fit_transform(bag_of_words)
tfidf_df = pd.DataFrame(tfidf.toarray(), columns=count_vectorizer.get_feature_names())
for i in range(6, 10):
    tfidf_df.loc[i][tfidf_df.loc[i] > 0].sort_values(ascending=False)

post_tfidf = pd.concat([data[['post_id']], tfidf_df], axis=1)
post_tfidf.to_csv('~/Desktop/tfidf.csv', index=False)

out_data = data[['_id', 'post_id', 'user_name', 'rating', 'content']]
out_data['len'] = out_data.content.str.len()
out_data.sort_values('len', ascending=False, inplace=True)
out_data = out_data.drop('len', axis=1).reset_index(drop=True)
out_data = out_data.loc[100:200]
out_data.shape
out_data.to_csv('~/Desktop/unlabbeled_1.csv', index=False, encoding='utf-8-sig')

data.to_csv('~/Desktop/all.csv', index=False)