import math
import string
from random import random

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import euclidean_distances
import numpy as np
from nltk.corpus import stopwords


def termfreq(document, word):
    N = len(document)
    occurance = len([token for token in document if token == word])
    return occurance/N

def idf(word, di, filter):
    try:
        word_occurance = di[word] + 1
    except:
        word_occurance = 1
    return np.log(len(filter)/word_occurance)


def tf_idf(di, filter):
    tf_idf_vec = np.zeros((len(filter)))
    for word in filter:
        tf1 = termfreq(filter, word)
        idf1 = idf(word, di, filter)
        value = tf1 * idf1
        tf_idf_vec[di[word]] = value
    return tf_idf_vec

def calc(names):
    arr = []
    stop_words = set(stopwords.words('english'))
    for name in names:
        with open(name, 'r') as f:
            data = f.read().lower().translate(str.maketrans('', '', string.punctuation)). \
                translate(str.maketrans('', '', string.digits)).split()
            arr.extend(data)


    filter = [word for word in arr if not word in stop_words]

    print(filter)
    di = {}
    for word in filter:
        if word in di:
            di[word] += 1
        if word not in di:
            di[word] = 1

    vectors = []

    vec = tf_idf(di, filter)
    vectors.append(vec)
    print(vectors[0])
    #
    # centroid_1 = [random.random() for i in range(6)]
    # centroid_2 = [random.random() for i in range(6)]
    # #
    # for centroid in [centroid_1, centroid_2]:
    #     print(math.dist(vectors[0], centroid))

    # X, y = load_iris(return_X_y=True)
    # km = KMeans(n_clusters=vectors[0], random_state=1).fit(X)
    # dists = euclidean_distances(km.cluster_centers_)
    # tri_dists = dists[np.triu_indices(5, 1)]
    # max_dist, avg_dist, min_dist = tri_dists.max(), tri_dists.mean(), tri_dists.min()



names = ['text_1.txt']
calc(names)
names = ['text_2.txt']
calc(names)
names = ['text_3.txt']
calc(names)
names = ['text_4.txt']
calc(names)
names = ['text_5.txt']
calc(names)
names = ['text_6.txt']
calc(names)
names = ['text_7.txt']
calc(names)
names = ['text_8.txt']
calc(names)
names = ['text_9.txt']
calc(names)
names = ['text_10.txt']
calc(names)


#Term Frequency
