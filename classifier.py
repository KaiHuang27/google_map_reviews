import jieba
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import cross_val_score
import pickle
import pymongo


def save_model(file_path, model):
    with open(file_path, 'wb') as f:
        pickle.dump(model, f)

def load_stopwords(file_path):
    with open(file_path, 'r') as f:
        return f.read().splitlines()


client = pymongo.MongoClient(host='mongodb://db:27017/')
db = client.google_map
data = pd.DataFrame(db.reviews.find({'content': {'$ne': 'null'}}))
data.drop_duplicates(subset='post_id', inplace=True, ignore_index=True)
data = data[~data.content.str.contains('由 Google 提供翻譯') & ~data.content.str.contains('null')].reset_index(drop=True)
#data = pd.read_csv('~/Desktop/all.csv')

labeled = pd.read_csv('~/Desktop/labeled_data.csv')
labeled_1 = pd.read_csv('~/Desktop/labeled_1.csv')
labeled_2 = pd.read_csv('~/Desktop/labeled_2.csv')
labeled_data = pd.concat([labeled, labeled_1, labeled_2], axis=0).drop_duplicates(subset='post_id').reset_index(drop=True)
labeled_data = labeled_data[['post_id', 'content', 'taste', 'service' ,'environment', 'price']]

# segmentation
data['content'] = data.content.str.replace('[\n，。]', ' ', regex=True)
data['segment'] = data['content'].apply(lambda x: jieba.lcut(x))

# bag of words
stopwords = load_stopwords('./stopwords.txt')
count_vectorizer = CountVectorizer(token_pattern=r'(?u)\b\w{2,10}\b', stop_words=stopwords)
bag_of_words = count_vectorizer.fit_transform(data.segment.str.join(' '))
#save_model('/Users/kai/Desktop/bow_model.pkl', count_vectorizer)

# tfidf
transformer = TfidfTransformer()
transformer.fit_transform(bag_of_words)
#save_model('/Users/kai/Desktop/tfidf_model.pkl', transformer)

# tfidf for train data
labeled_data['segment'] = labeled_data.content.apply(lambda x: jieba.lcut(x))
labeled_data_bow = count_vectorizer.transform(labeled_data.segment.str.join(' '))
train_bow = pd.DataFrame(labeled_data_bow.toarray(), columns=count_vectorizer.get_feature_names())
train_tfidf = transformer.transform(labeled_data_bow)
train_tfidf = pd.DataFrame(train_tfidf.toarray(), columns=count_vectorizer.get_feature_names())


# prepare training set and testing set
target_cols = ['taste', 'service', 'environment', 'price']

print(labeled_data[target_cols].sum())
print(labeled_data[target_cols].mean())

bayes = MultinomialNB(fit_prior=True, class_prior=None)
logistic = LogisticRegression(max_iter=1000)
svc = LinearSVC()
classifiers = [bayes, logistic, svc]
features = ['tfidf', 'bow']

label = labeled_data[target_cols].astype(int)
all_pred = []

for clf in classifiers:
    print(f'Model: {clf.__class__.__name__}')
    classifier = OneVsRestClassifier(clf, n_jobs=-1)

    for feature in features:
        if feature == 'tfidf':
            train = train_tfidf
        elif feature == 'bow':
            train = train_bow

        x_train, x_test, y_train, y_test = train_test_split(train, label, random_state=3)

        print(f'Feature: {feature}')
        print('Training...')
        classifier.fit(x_train, y_train)
        prediction = classifier.predict(x_test)

        for i, category in enumerate(target_cols):
            print('{}\nAccuracy: {}\nF1-score: {}'.format(
                category,
                round(accuracy_score(y_test[category], prediction.T[i]), 6),
                round(f1_score(y_test[category], prediction.T[i]), 6)
            ))

        all_pred.append(classifier.predict(train))

# stacking model
for i, category in enumerate(target_cols):
    print(f'\n{category.upper()} stacking model')
    stack_train = np.vstack([p.T[i] for p in all_pred]).T
    x_train, x_test, y_train, y_test = train_test_split(stack_train, label[category], train_size=0.5, random_state=3)
    
    classifier = LogisticRegression(max_iter=1000)
    classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)
    print(f'Accuracy: {round(accuracy_score(y_test, pred), 6)}')
    print(f'F1-score: {round(f1_score(y_test, pred), 6)}')

    x, y = stack_train, label[category]
    k = 10
    scores = cross_val_score(classifier, x, y ,cv=k, scoring='accuracy')
    print(f'{k}-fold CV Accuracy: {scores.mean()}')
    scores = cross_val_score(classifier, x, y ,cv=k, scoring='f1')
    print(f'{k}-fold CV F1-score: {scores.mean()}')

'''
final_pred = final_pred / (len(classifiers) * len(features))
final_pred = (final_pred > 0.5).astype(int)
print(f'Final Accuracy: {round(accuracy_score(y_test, final_pred), 6)}')
'''
# train first level model
trained_clf = []
for clf in classifiers:
    clf_name = clf.__class__.__name__
    print(f'Model: {clf_name}')
    classifier = OneVsRestClassifier(clf, n_jobs=-1)

    for feature in features:
        if feature == 'tfidf':
            train = train_tfidf
        elif feature == 'bow':
            train = train_bow

        print(f'Feature: {feature}')

        print('Cross Validation...')
        k = 5
        f1 = cross_val_score(classifier, train, label ,cv=k, scoring='f1_micro')
        print(f'{k}-fold CV micro F1-score: {f1.mean()}')

        print('Training...')
        classifier.fit(train, label)
        
        print(f'Save Model: {clf_name}, Feature: {feature}')
        trained_clf.append({'model': classifier, 'feature': feature})

save_model('/Users/kai/Desktop/first_level_model.pkl', trained_clf)


# train second level model
trained_second_models = {}
for i, category in enumerate(target_cols):
    print(f'\n{category.upper()} second level model')
    stack_train = np.vstack([p.T[i] for p in all_pred]).T
    x_train, x_test, y_train, y_test = train_test_split(stack_train, label[category], train_size=0.5, random_state=3)
    
    classifier = LogisticRegression(max_iter=1000)
    '''
    classifier.fit(x_train, y_train)
    pred = classifier.predict(x_test)
    print(f'Accuracy: {round(accuracy_score(y_test, pred), 6)}')
    print(f'F1-score: {round(f1_score(y_test, pred), 6)}')
    '''
    x, y = stack_train, label[category]
    k = 10
    scores = cross_val_score(classifier, x, y ,cv=k, scoring='accuracy')
    print(f'{k}-fold CV Accuracy: {scores.mean()}')
    scores = cross_val_score(classifier, x, y ,cv=k, scoring='f1')
    print(f'{k}-fold CV F1-score: {scores.mean()}')
    
    classifier.fit(x, y)
    
    trained_second_models[category] = classifier

save_model('/Users/kai/Desktop/second_level_model.pkl', trained_second_models)
