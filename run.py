
import numpy as np
import os
from tqdm.notebook import tqdm
from eeglib.helpers import EDFHelper
from sklearn.decomposition import PCA
import networkx as nx
from node2vec import Node2Vec
data_path = "data"
# find all files in the data directory
files = os.listdir(data_path)
files = sorted(files)
count = 0
stressed = []
not_stressed = []

# x is a pd dataframe with the features

# gt is a list of booleans, true if stressed, false if not
W = 2500
dataset=[]
S = 35
for t in tqdm(range(0, S)):
    print(t)
    unstressed_file = "Subject"+str(t).rjust(2, '0')+"_1.edf"
    stressed_file = "Subject"+str(t).rjust(2, '0')+"_2.edf"
    # read the edf file
    gt=[]
    x=[]
    is_stressed = False
    all_files = []
    vectors = []
    matrices = []
    for file in [unstressed_file, stressed_file]:
        helper= EDFHelper("data/"+str(file), sampleRate=500, lowpass=30, highpass=0.5, windowSize=W)
        cnt = 0
        for eeg in helper:
            cnt += 1
            print(cnt)
            # for this particular window
            dat = eeg.getChannel()
            # create N by N matrix

            l = len(dat)
            mat = [[0 for i in range(l)] for j in range(l)]


            for i in range(l):
                for j in range(l):
                    # calculate pearson's correlation between dat[i] and dat [j]
                    # use eeg CCC
                    mat[i][j] = 1 - eeg.DTW(channels=[i, j])[0]
                    #mat[i][j] = 1 - abs(np.(dat[i], dat[j])[0][1]) # pearson's correlation coefficient distance
            '''
            # floyd warshall to find the shortest path HAHAHAHAHHA LOL HAHAHAHAHAHAHAHHA
            for k in range(l):
                for i in range(l):
                    for j in range(l):
                        if i != j:
                            mat[i][j] = min(mat[i][j], mat[i][k] + mat[k][j])
            '''
            G = nx.Graph()
            for i in range(l):
                G.add_node(i)

            for i in range(l):
                for j in range(l):
                    if i != j:
                        #print(i,j,mat[i][j])
                        G.add_edge(i, j, weight=mat[i][j])

            node2vec = Node2Vec(G, dimensions=10, walk_length=16, num_walks=100, workers=8, quiet=True)
            # TODO: remember to add more dimensions
            model = node2vec.fit()
            vecs = model.wv
            vecs = np.array([vecs[i] for i in range(l)])
            vecs = vecs.flatten(order='C')

            vectors.append(vecs)
            gt.append(is_stressed)
        is_stressed = True

    dataset.append((vectors, gt))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
classifiers = [
    RandomForestClassifier(),
    SVC(kernel="linear", C=0.025),
    KNeighborsClassifier(3),
    #GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
]
for classifier in classifiers:

    accuracies = []
    for person in dataset:
        x = np.array(person[0])
        gt = person[1]
        # convert x from a list to a pd dataframe

        # convert gt from a list to a pd dataframe
        x_train, x_test, y_train, y_test = train_test_split(x, gt, test_size=0.2)
        # convert the dataframes to numpy arrays
        # create the vectorizer
        # fit the vectorizer to the training data
        # scale the training data
        '''
        pca_30 = PCA(n_components=30, random_state=69)
        x_train = pca_30.fit_transform(x_train)
        x_test = pca_30.transform(x_test)
        '''

        clf = classifier
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        #print(y_test, y_pred)

        accuracies.append(accuracy_score(y_test, y_pred))

    #print(accuracies)
    print(f"Average Accuracy of {classifier}", np.mean(accuracies))
    print(accuracies)