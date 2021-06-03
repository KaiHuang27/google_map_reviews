import pickle
from bson.py3compat import reraise
import jieba
import pandas as pd
import numpy as np
import pymongo
from pymongo import UpdateOne
from bson.objectid import ObjectId


jieba.load_userdict("userdict.txt")
client = pymongo.MongoClient(host='mongodb://db:27017/')
db = client.google_map


def load_model(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def load_data(name):
    nids = [q['nid'] for q in db.places.find({'name': {'$regex': name}}, {'_id':0, 'nid':1})]
    data = pd.DataFrame(db.reviews.find({'nid': {'$in': nids}, 'tags': {'$eq': None}}))
    if data.empty:
        return None
    else:
        return data.set_index('_id')

def text_segmentation(text_in_series):
    return text_in_series.apply(lambda x: jieba.lcut(x))

def get_bag_of_words(segment):
    c_vectorizer = load_model('./models/bow_model.pkl')
    return pd.DataFrame(c_vectorizer.transform(segment.str.join(' ')).toarray(),
                        columns=c_vectorizer.get_feature_names(),
                        index=segment.index)

def get_tfidf(bag_of_words):
    tfidf_transformer = load_model('./models/tfidf_model.pkl')
    return pd.DataFrame(tfidf_transformer.transform(bag_of_words).toarray(),
                        columns=bag_of_words.columns,
                        index=bag_of_words.index)
# keyword
def save_keywords(bag_of_words, tf_idf):
    requests = []
    for i in bag_of_words.index:
        keywords = list(set(bag_of_words.loc[i][bag_of_words.loc[i] > 0].sort_values(ascending=False)[:3].index.tolist() + \
                            tf_idf.loc[i][tf_idf.loc[i] > 0].sort_values(ascending=False)[:3].index.tolist()))
        requests.append(UpdateOne({'_id': ObjectId(i)}, {'$addToSet': {'keywords': {'$each': keywords}}}))
    db.reviews.bulk_write(requests)

def tagging(bag_of_words, tf_idf):
    first_level_models = load_model('./models/first_level_model.pkl')
    second_level_models = load_model('./models/second_level_model.pkl')

    # first level
    first_level_predictions = []
    for model in first_level_models:
        classifier = model['model']
        if model['feature'] == 'tfidf':
            x = bag_of_words
        elif model['feature'] == 'bow':
            x = tf_idf

        first_level_predictions.append(classifier.predict(x))

    # second level model
    label_catetgory = ['taste', 'service', 'environment', 'price']
    out = {}
    for i, cat in enumerate(label_catetgory):
        second_level_x = np.vstack([p.T[i] for p in first_level_predictions]).T
        classifier = second_level_models[cat]
        out[cat] = classifier.predict(second_level_x)

    return pd.DataFrame(out, index=bag_of_words.index)

# update reviews (tagging)
def save_tags(tags):
    requests = [UpdateOne({'_id': ObjectId(i)}, {'$addToSet': {'tags': {'$each': t[t == 1].index.tolist()}}}) for i, t in tags.iterrows()]
    db.reviews.bulk_write(requests)


if __name__ == '__main__':
    data = load_data('王品牛排')
    if not data.empty:
        data['segment'] = text_segmentation(data.content)
        bow = get_bag_of_words(data.segment)
        tfidf = get_tfidf(bow)
        save_keywords(bow, tfidf)
        tags = tagging(bow, tfidf)
        save_tags(tags)
