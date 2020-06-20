import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
# Setting display rows to to see more tuples
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn import preprocessing

scaler = MinMaxScaler()

pd.set_option('display.max_rows', 800)
os.getcwd()
print('Import of the packages Successful')
# Water Treatment Dataset
water_treatment_dataset = pd.read_csv("water-treatment.data", sep=",", header=None)
# Adding to a Data frame
df = pd.DataFrame(water_treatment_dataset)
df.columns = ['Date', 'Q-E', 'ZN-E', 'PH-E', 'DBO-E', 'DQO-E', 'SS-E', 'SSV-E', 'SED-E', 'COND-E', 'PH-P', 'DBO-P',
              'SS-P', 'SSV-P', 'SED-P', 'COND-P', 'PH-D', 'DBO-D', 'DQO-D', 'SS-D', 'SSV-D', 'SED-D', 'COND-D', 'PH-S',
              'DBO-S', 'DQO-S', 'SS-S', 'SSV-S', 'SED-S', 'COND-S', 'RD-DBO-P', 'RD-SS-P', 'RD-SED-P', 'RD-DBO-S',
              'RD-DQO-S', 'RD-DBO-G', 'RD-DQO-G', 'RD-SS-G', 'RD-SED-G']
# Replace ? with NaN values
data_frame = df.replace('?', np.nan)
#print(data_frame)
# Converting following Data frame features to numeric
data_frame[
    ['Q-E', 'ZN-E', 'PH-E', 'DBO-E', 'DQO-E', 'SS-E', 'SSV-E', 'SED-E', 'COND-E', 'PH-P', 'DBO-P', 'SS-P', 'SSV-P',
     'SED-P', 'COND-P', 'PH-D', 'DBO-D', 'DQO-D', 'SS-D', 'SSV-D', 'SED-D', 'COND-D', 'PH-S', 'DBO-S', 'DQO-S', 'SS-S',
     'SSV-S', 'SED-S', 'COND-S', 'RD-DBO-P', 'RD-SS-P', 'RD-SED-P', 'RD-DBO-S', 'RD-DQO-S', 'RD-DBO-G', 'RD-DQO-G',
     'RD-SS-G', 'RD-SED-G']] = data_frame[
    ['Q-E', 'ZN-E', 'PH-E', 'DBO-E', 'DQO-E', 'SS-E', 'SSV-E', 'SED-E', 'COND-E', 'PH-P', 'DBO-P', 'SS-P', 'SSV-P',
     'SED-P', 'COND-P', 'PH-D', 'DBO-D', 'DQO-D', 'SS-D', 'SSV-D', 'SED-D', 'COND-D', 'PH-S', 'DBO-S', 'DQO-S', 'SS-S',
     'SSV-S', 'SED-S', 'COND-S', 'RD-DBO-P', 'RD-SS-P', 'RD-SED-P', 'RD-DBO-S', 'RD-DQO-S', 'RD-DBO-G', 'RD-DQO-G',
     'RD-SS-G', 'RD-SED-G']].apply(pd.to_numeric)
# Testing Data types and NaN values
#print(data_frame.dtypes)
column_names = data_frame.columns
#print(column_names)
#print(data_frame['RD-DBO-G'].isnull())

# Creating list of features in the Dataset
column_list = ['Date', 'Q-E', 'ZN-E', 'PH-E', 'DBO-E', 'DQO-E', 'SS-E', 'SSV-E', 'SED-E', 'COND-E', 'PH-P', 'DBO-P',
               'SS-P', 'SSV-P', 'SED-P', 'COND-P', 'PH-D', 'DBO-D', 'DQO-D', 'SS-D', 'SSV-D', 'SED-D', 'COND-D', 'PH-S',
               'DBO-S', 'DQO-S', 'SS-S', 'SSV-S', 'SED-S', 'COND-S', 'RD-DBO-P', 'RD-SS-P', 'RD-SED-P', 'RD-DBO-S',
               'RD-DQO-S', 'RD-DBO-G', 'RD-DQO-G', 'RD-SS-G', 'RD-SED-G']

# Filling the NaN values with Median
i = 1
while i < len(column_list):
    median = data_frame[column_list[i]].median()
    data_frame[column_list[i]].fillna(median, inplace=True)
    i += 1
#print(data_frame)
print('Null Values cleaned sucessfully')

# Dropping Date Column for the Dataset to be normalized
data_frame.drop(data_frame.columns[0], axis=1, inplace=True)
#print(data_frame)
print('Dropped Date Column from Dataset')

# Normalizing the Data
pd.set_option('display.max_rows', 800)
#x = data_frame.values
#data_frame_normalized = preprocessing.normalize(x)
data_frame_normalized = normalize(data_frame, axis=0, norm='max')
#print(data_frame_normalized)
print('Data sucessfully normalized')

# Elbow Graph method to determine optimum number of clusters
elbow = []
for i in range(1, 10):
    Kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    Kmeans.fit(data_frame_normalized)
    elbow.append(Kmeans.inertia_)
plt.plot(range(1, 10), elbow, 'bx-')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
#print(elbow)
print('Printed Elbow graph')

# Applying Modified K-means on Water Treatment Dataset for printing
Kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = Kmeans.fit_predict(data_frame_normalized)
#print(y_kmeans)
outF = open("KmeansOutput.txt", "w")
i = 1
for i in range(1, 528):
    outF.write(str(i))
    outF.write("\t")
    outF.write(str(y_kmeans[i - 1]+1))
    outF.write("\n")
    i = + 1
outF.close()
print('Printed the clusters to the file ')

# print(data_frame_normalized)

plt.scatter(data_frame_normalized[y_kmeans == 0, 0], data_frame_normalized[y_kmeans == 0, 1], s=100, c='red',
            label='Cluster 1')
plt.scatter(data_frame_normalized[y_kmeans == 1, 0], data_frame_normalized[y_kmeans == 1, 1], s=100,
            c='green',
            label='Cluster 2')
plt.scatter(data_frame_normalized[y_kmeans == 2, 0], data_frame_normalized[y_kmeans == 2, 1], s=100,
            c='purple',
            label='Cluster 3')
plt.scatter(data_frame_normalized[y_kmeans == 3, 0], data_frame_normalized[y_kmeans == 3, 1], s=100, c='cyan',
            label='Cluster 4')
plt.scatter(data_frame_normalized[y_kmeans == 4, 0], data_frame_normalized[y_kmeans == 4, 1], s=100, c='Yellow',
            label='Cluster 5')
#plt.scatter(Kmeans.cluster_centers_[:, 0], Kmeans.cluster_centers_[:, 1], s=300, c='Blue', label='Centroids')
plt.title('Clusters')
plt.legend()
plt.show()

# Calculating eigenvectors and eigenvalues on covariance matrix
sc = StandardScaler()
X_std = sc.fit_transform(data_frame_normalized)
mean_vec = np.mean(X_std, axis=0)
cov_mat = (X_std - mean_vec).T.dot((X_std - mean_vec)) / (X_std.shape[0] - 1)
#print('Covariance matrix n%s' % cov_mat)
cov_mat = np.cov(X_std.T)
eig_vals, eig_vecs = np.linalg.eig(cov_mat)
#print('Eigenvectors n%s' % eig_vecs)
#print('nEigenvalues n%s' % eig_vals)
eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:, i]) for i in range(len(eig_vals))]
#print('Eigenvalues in descending order:')
#for i in eig_pairs:
 #   print(i[0])

# PCA on Water Treatment Dataset
X = data_frame_normalized
X_std = StandardScaler().fit_transform(X)
#print(X_std)
pca = PCA(n_components=5)
X_transform = pca.fit_transform(X_std)
#print(pca.explained_variance_ratio_)
#print(X_transform)
#print(X_transform.shape)
j = 1
for i in range(3):
    plt.scatter(X_transform[:, i], X_transform[:, j])
    i += 1
    j += 1
plt.scatter(X_transform[:, 4], X_transform[:, 0])
plt.title("PCA for Water Treatment Dataset")
plt.show()

# Elbow Graph method to determine optimum number of clusters after PCA Applied
elbow_pca = []
for i in range(1, 10):
    Kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
    Kmeans.fit(X_transform)
    elbow_pca.append(Kmeans.inertia_)
plt.plot(range(1, 10), elbow_pca, 'bx-')
plt.title('Elbow Method after PCA')
plt.xlabel('Number of clusters')
plt.ylabel('Distortion')
plt.show()
print(elbow_pca)

# Applying Modified K-means on Water Treatment Dataset after PCA
Kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = Kmeans.fit_predict(X_transform)
#print(y_kmeans)
outF = open("KmeansPCAOutput.txt", "w")
i = 1
for i in range(1, 528):
    outF.write(str(i))
    outF.write("\t")
    outF.write(str(y_kmeans[i - 1]+1))
    outF.write("\n")
    i = + 1
outF.close()

print('After PCA: Printed the clusters to the file')

plt.scatter(X_transform[y_kmeans == 0, 0], X_transform[y_kmeans == 0, 1], s=100, c='red',
            label='Cluster 1')
plt.scatter(X_transform[y_kmeans == 1, 0], X_transform[y_kmeans == 1, 1], s=100,
            c='green',
            label='Cluster 2')
plt.scatter(X_transform[y_kmeans == 2, 0], X_transform[y_kmeans == 2, 1], s=100,
            c='purple',
            label='Cluster 3')
plt.scatter(X_transform[y_kmeans == 3, 0], X_transform[y_kmeans == 3, 1], s=100, c='cyan',
            label='Cluster 4')
plt.scatter(X_transform[y_kmeans == 4, 0], X_transform[y_kmeans == 4, 1], s=100, c='Yellow',
            label='Cluster 5')
plt.scatter(Kmeans.cluster_centers_[:, 0], Kmeans.cluster_centers_[:, 1], s=300, c='Blue', label='Centroids')
plt.title('Clusters')
plt.legend()
plt.show()