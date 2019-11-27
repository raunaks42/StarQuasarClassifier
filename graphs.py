import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

# Load the example Titanic dataset
#titanic = sns.load_dataset("titanic")
df=pd.read_csv('tottimes.txt')
# Draw a nested barplot to show survival for class and sex
g = sns.catplot(x="model", y="tottime", data=df,
                height=6, kind="bar", palette="muted")
g.despine(left=True)
g.set_ylabels("Total Execution Time")
plt.xticks(
    rotation=25, 
    horizontalalignment='right',
    fontweight='light',
    fontsize='x-small'
)
g.set_xlabels('Model')
#plt.ylim(75,100)
plt.savefig('tottime.png')