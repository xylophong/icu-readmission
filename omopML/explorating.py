import seaborn as sns
import numpy as np

def distplot_with_hue(data=None, x=None, hue=None, row=None, col=None, 
                      height=None, aspect=1, legend=True, **kwargs):
    _, bins = np.histogram(data[x].dropna())
    g = sns.FacetGrid(data, hue=hue, row=row, col=col, height=height, aspect=aspect)
    g.map(sns.distplot, x, **kwargs)
