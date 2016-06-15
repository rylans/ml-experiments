'''4.2 feature extraction

http://scikit-learn.org/stable/modules/feature_extraction.html
'''

def features_from_dicts():
    '''4.2.1 loading features from dicts'''

    measurements = [
        {'city': 'dubai', 'temperature': 33.},
        {'city': 'london', 'temperature': 12.},
        {'city': 'san francisco', 'temperature': 18.}]

    from sklearn.feature_extraction import DictVectorizer
    vec = DictVectorizer()

    print vec.fit_transform(measurements).toarray()
    print vec.get_feature_names()

def tfidf_term():
    '''4.2.3.4 Tf-idf term weighting'''

    from sklearn.feature_extraction.text import TfidfTransformer
    transformer = TfidfTransformer()
    print transformer

    counts = [[3, 0, 1],
              [2, 0, 0],
              [3, 0, 0],
              [4, 0, 0],
              [3, 2, 0],
              [3, 0, 2]]

    tfidf = transformer.fit_transform(counts)
    print tfidf.toarray()

if __name__ == '__main__':
    features_from_dicts()
    tfidf_term()
