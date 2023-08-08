# import required packages
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# example data
X = np.array([[1,4,4], [2,6,6], [3,10,16], [4,5,4], [5,8,12], [6,7,5], [7,11,8], [8,12,10]])

# perform preprocessing of features
pca = PCA(n_components=1)

# fit the data running the PCA algorithm - automatically performs mean normalisation in the pca function
pca.fit(X)

# examine how much variance is explained by each PC - explained_variance_ratio
var_exp = pca.explained_variance_ratio_

# transform the data onto new axes
X_transformed = pca.transform(X)

# reconstruction of original data
X_reduced = pca.inverse_transform(X_transformed)

# correlate PCs with one another
pc_corr = pd.DataFrame(X_transformed).corr().stack().sort_values()