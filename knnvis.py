import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets, neighbors
from mlxtend.plotting import plot_decision_regions

def knn_comparison(data, k):
    x = data[['principal component 1','principal component 2']].values
    y = data['class'].astype(int).values
    clf = neighbors.KNeighborsClassifier(n_neighbors=k)
    clf.fit(x, y)# Plotting decision region
    plot_decision_regions(x, y, clf=clf, legend=2)# Adding axes annotations
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Knn with K='+ str(k))
    plt.savefig('knnvis'+repr(k)+'.png')

data1 = pd.read_csv('pca_cat1.csv')
knn_comparison(data1, 25)