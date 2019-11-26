import numpy as np
# import matplotlib.pyplot as plt
import seaborn as sns
import os
import pandas as pd
import matplotlib.pyplot as plt

direc=os.fsencode('.')  #current directory
for File in os.listdir(direc):  #each file in directory
    filename=os.fsdecode(File)  #get filename
    if (filename.endswith('.csv') and filename.startswith('pca_')):   #only if csv
        df= pd.read_csv(filename)
        # x = df['principal component 1']
        # y = df['principal component 2']
        # plt.scatter(x, y, edgecolors='r')
        # plt.xlabel('x')
        # plt.ylabel('y')
        # plt.title(filename)
        # plt.show()
        sns.lmplot('principal component 1', 'principal component 2', df, hue='class', fit_reg=False)
        
        fig = plt.gcf()
        # fig.set_size_inches(15, 10)
        plt.title(filename)
        plt.show()