import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from matplotlib import pyplot
from statsmodels.graphics.gofplots import qqplot
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA
from scipy.fftpack import dct

print('Import of Packages Successful')
os.getcwd()
# First DataSet
dataset1_1 = pd.read_csv("dist1_500_1.txt", sep=" ", header=None)
dataset1_2 = pd.read_csv("dist1_500_2.txt", sep=" ", header=None)
Dataframe1_1 = pd.DataFrame(dataset1_1)
Dataframe1_2 = pd.DataFrame(dataset1_2)
Dataframe1_1 = Dataframe1_1.dropna(how='all')
Dataframe1_2 = Dataframe1_2.dropna(how='all')
dataFrameComb1 = pd.concat([Dataframe1_1, Dataframe1_2], sort=True)
print(type(dataFrameComb1))
print("Printing Dataframe 1")
print(dataFrameComb1)
print("Dataframe 1 shape")
print(dataFrameComb1.shape)
Data_samples1 = dataFrameComb1.sample(n=10)
print('Printing 10 samples from Dataset 1')
print(Data_samples1)
# Plotting the 10 Samples from Dataset 1 using different plotting techniques
plt.boxplot(Data_samples1)
plt.title('Box Plot for DataSet 1')
qqplot(Data_samples1, line='s')
plt.title('QQ Plot for dataset 1')
pyplot.show()
pyplot.hist(Data_samples1,density='true')
plt.title('Histogram for DataSet 1')
pyplot.show()

sc = StandardScaler()
X_std = sc.fit_transform(dataFrameComb1)
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0] - 1)
print('Covariance matrix n%s' % cov_mat)

# Calculating eigenvectors and eigenvalues on covariance matrix

cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
print('Eigenvectors n%s' % eig_vecs)
print('nEigenvalues n%s' % eig_vals)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
print('Eigenvalues in descending order:')
for i in eig_pairs:
    print(i[0])

# Principle Component Analysis
X = dataFrameComb1
X_std = StandardScaler().fit_transform(X)
print(X_std)
pca = PCA(n_components=71)
X_transform = pca.fit_transform(X_std)
print(pca.explained_variance_ratio_)
print(X_transform)
print(X_transform.shape)
j = 1
for i in range(69):
    plt.scatter(X_transform[:, i], X_transform[:, j])
    i += 1
    j += 1
plt.scatter(X_transform[:, 70], X_transform[:, 0])
plt.title("PCA for DataSet 1")
plt.show()

# Discrete Cosine Transform

print(X_std)


def dct1(X_std):
    return dct(dct(X_std.T, norm='ortho').T, norm='ortho')


s = dct1(X_std)
print(s.shape)
print(s)
j = 1
for i in range(98):
    plt.scatter(s[:, i], s[:, j])
    i += 1
    j += 1
plt.scatter(s[:, 99], s[:, 0])
plt.title("DCT for DataSet 1")
plt.show()

# Independent Component Analysis

transformer = FastICA(n_components=71, random_state=0)
X_transformed = transformer.fit_transform(X)
fig = plt.figure()
print(X_transformed.shape)
j = 1
for i in range(69):
    plt.scatter(X_transformed[:, i], X_transformed[:, j])
    i += 1
    j += 1
plt.scatter(X_transformed[:, 70], X_transformed[:, 0])
plt.title("ICA for DataSet 1")
plt.show()
print(X_transformed)
print(40 * '_')

# Second DataSet
dataset2_1 = pd.read_csv("dist2_500_1.txt", sep=" ", header=None)
dataset2_2 = pd.read_csv("dist2_500_2.txt", sep=" ", header=None)

Dataframe2_1 = pd.DataFrame(dataset2_1)
Dataframe2_2 = pd.DataFrame(dataset2_2)
Dataframe2_1 = Dataframe2_1.dropna(how='all')
Dataframe2_2 = Dataframe2_2.dropna(how='all')
dataFrameComb2 = pd.concat([Dataframe2_1, Dataframe2_2], sort=True)
print(type(dataFrameComb2))
print("Printing Dataframe 2")
print(dataFrameComb2)
Data_samples2 = dataFrameComb2.sample(n=10)
# Plotting the 10 Samples from Dataset 2 using different plotting techniques
plt.boxplot(Data_samples2)
plt.title('Box Plot for DataSet 2')
qqplot(Data_samples2, line='s')
plt.title('QQ Plot for DataSet 2')
pyplot.show()
pyplot.hist(Data_samples2)
plt.title('Histogram for DataSet 2')
pyplot.show()

sc = StandardScaler()
X_std2 = sc.fit_transform(dataFrameComb2)
mean_vec2 = np.mean(X_std2, axis=0)
cov_mat2 = (X_std2 - mean_vec2).T.dot((X_std2 - mean_vec2)) / (X_std2.shape[0] - 1)
print('Covariance matrix n%s' % cov_mat2)

# Calculating eigenvectors and eigenvalues on covariance matrix for Dataset 2

cov_mat2 = np.cov(X_std2.T)
eig_vals2, eig_vecs2 = np.linalg.eig(cov_mat2)
print('Eigenvectors n%s' % eig_vecs2)
print('nEigenvalues n%s' % eig_vals2)
eig_pairs2 = [(np.abs(eig_vals2[i]), eig_vecs2[:, i]) for i in range(len(eig_vals2))]
print('Eigenvalues in descending order:')
for i in eig_pairs2:
    print(i[0])

# Principle Component Analysis (PCA)
X2 = dataFrameComb2
X_std2 = StandardScaler().fit_transform(X2)
pca = PCA(n_components=71)
X_transform2 = pca.fit_transform(X_std2)
print(pca.explained_variance_ratio_)
print(X_transform2)
print(X_transform2.shape)
j = 1
for i in range(69):
    plt.scatter(X_transform2[:, i], X_transform2[:, j])
    i += 1
    j += 1
plt.scatter(X_transform2[:, 70], X_transform2[:, 0])
plt.title("PCA for DataSet 2")
plt.show()

# Discrete Cosine Transform
def dct2(X_std2):
    return dct(dct(X_std2.T, norm='ortho').T, norm='ortho')

r = dct2(X_std2)
print(r.shape)
print(r)
j = 1
for i in range(98):
    plt.scatter(r[:, i], r[:, j])
    i += 1
    j += 1
plt.scatter(r[:, 99], r[:, 0])
plt.title("DCT for DataSet 2")
plt.show()

# Independent Component Analysis(ICA)
transformer2 = FastICA(n_components=71, random_state=0)
X_transformed2 = transformer2.fit_transform(X2)
fig = plt.figure()
print(X_transformed2.shape)
j = 1
for i in range(69):
    plt.scatter(X_transformed2[:, i], X_transformed2[:, j])
    i += 1
    j += 1
plt.scatter(X_transformed2[:, 70], X_transformed2[:, 0])
plt.title("ICA for DataSet 2")
plt.show()
print(X_transformed2)