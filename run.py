import pandas as pd
import numpy as np
import os
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
W = 500
overlap = 250
dataset=[]
S = 36
for t in range(0,S):
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

        for eeg in helper:
            # for this particular window
            dat = eeg.getChannel()
            # create N by N matrix

            l = len(dat)
            mat = [[0 for i in range(l)] for j in range(l)]
            for i in range(l):
                for j in range(l):
                    # calculate pearson's correlation between dat[i] and dat [j]
                    mat[i][j] = np.corrcoef(dat[i], dat[j])[0][1]
            matrices.append(mat)
            gt.append(is_stressed)
        is_stressed = True

    # convert the list of matrices to a list of features
    nw = len(matrices)
    B = [[0 for i in range(nw)] for j in range(nw)]
    # create a new nx graph
    G = nx.Graph()
    # create nodes in the graph
    for i in range(nw):
        G.add_node(i)
    for i in range(nw):
        for j in range(nw):
            if i != j:
                # calculate the similarity between matrices[i] and matrices[j]
                G.add_edge(i, j, weight=np.linalg.norm(np.array(matrices[i]) - np.array(matrices[j])))
    # create a node2vec object
    node2vec = Node2Vec(G, dimensions=90, walk_length=16, num_walks=100, workers=8)
    # train node2vec
    model = node2vec.fit()
    # get the vectors
    vecs = model.wv
    # convert the vectors to a list
    vecs = [vecs[i] for i in range(nw)]
    vectors.extend(vecs)

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
    GaussianNB(),
    QuadraticDiscriminantAnalysis()
]
for classifier in classifiers:

    accuracies = []
    for person in dataset:
        x=person[0]
        gt=person[1]
        x=pd.DataFrame(x)
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

        accuracies.append(accuracy_score(y_test, y_pred))

    #print(accuracies)
    print(f"Average Accuracy of {classifier}", np.mean(accuracies))
'''
Average Accuracy of RandomForestClassifier() 0.7412364848115817
Average Accuracy of SVC(C=0.025, kernel='linear') 0.7450804880523987
Average Accuracy of KNeighborsClassifier(n_neighbors=3) 0.6895356381730976
Average Accuracy of DecisionTreeClassifier() 0.6603325331110679
Average Accuracy of MLPClassifier(alpha=1, max_iter=1000) 0.7289350472397322
Average Accuracy of AdaBoostClassifier() 0.7055369724314278
Average Accuracy of GaussianNB() 0.7023448508774995
Average Accuracy of QuadraticDiscriminantAnalysis() 0.7393483345893994

'''