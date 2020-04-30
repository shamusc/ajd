import scipy
import pandas as pd
import numpy as np
from sklearn import manifold
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
import umap
import random
import os
import sys
import csv

def neighbors(data, k=20):
    # for a given dataset, finds the k nearest neighbors for each point
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(data)
    distances, indices = nbrs.kneighbors(data)
    return indices[:,1:]

def jaccard(A,B):
    # for two sets A and B, finds the Jaccard distance J between A and B
    A = set(A)
    B = set(B)
    union = list(A|B)
    intersection = list(A & B)
    J = ((len(union) - len(intersection))/(len(union)))
    return(J)


def ndr(data,method,dim,n_neighbors=100):
    # Given a dataset, this function performs dimensionality reduction into the
    # dimension specified by dimwith the specified method.
    if method == 'standard_LLE':
        embedding = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors,n_components=dim,\
                method='standard').fit_transform(data)
    elif method == 'hessian_LLE':
        embedding = manifold.LocallyLinearEmbedding(n_neighbors=235,n_components=dim,\
                method='hessian',eigen_solver='dense').fit_transform(data)
    elif method == 'ltsa_LLE':
        embedding = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=dim,\
                method='ltsa',eigen_solver='dense').fit_transform(data)
    elif method == 'modified_LLE':
        embedding = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=dim,\
                method='modified',eigen_solver='dense').fit_transform(data)
    elif method == 'IsoMap':
        embedding = manifold.Isomap(n_neighbors=n_neighbors, n_components=dim)\
            .fit_transform(data)
    elif method == 't-SNE':
        embedding = manifold.TSNE(n_components=dim, init='pca', random_state=0,method='exact')\
                .fit_transform(data)
    elif method == 'MDS':
        embedding = manifold.MDS(n_components=dim, max_iter=100, n_init=1).fit_transform(data)
    elif method == 'Spectral_Embedding':
        embedding = manifold.SpectralEmbedding(n_components=dim,n_neighbors=n_neighbors,eigen_solver= 'arpack' )\
                .fit_transform(data)
    elif method == 'UMAP':
        embedding = umap.UMAP(n_components=dim,n_neighbors=n_neighbors).fit_transform(data)
    elif method == 'PCA':
        embedding = PCA(n_components=dim,svd_solver= 'auto').fit_transform(data)
    embedding = pd.DataFrame(embedding)
    return(embedding)

def ajd(data,embedding,jnn=20):
    # Given a dataset and its lower dimensional representation, this function
    # finds the average jaccard distaqnce between the two.
    print("Finding High-D Neighborhood")
    high_D_neighborhood = neighbors(data,k=jnn)
    print("Finding Low-D Neighborhood...")
    low_D_neighborhood = neighbors(embedding,k=jnn)
    print("Calculating Jaccard Distances...")
    jaccard_distances = 0
    n_samples = data.shape[0]
    i = 0
    while i < n_samples:
        jaccard_distance = jaccard(low_D_neighborhood[i,:],high_D_neighborhood[i,:])
        jaccard_distances = jaccard_distances + jaccard_distance
        i += 1
    result = jaccard_distances/n_samples
    return result

def hypersphere(n_dimensions,n_samples=1000,k_space=20,section=False,offset=0,\
    offset_dimension=0,noise=False,noise_amplitude=.01):
    # This function creates a hyperspherical dataset of arbitrary dimension
    # by sampling from a hypersphere of radius one.
    random.seed()
    data = np.zeros((n_samples,k_space))
    i = 0
    while i < n_samples:
        j = 0
        while j < n_dimensions:
            if section == True:
                a = random.random()
            else:
                a = random.uniform(-1,1)
            data[i,j]=a
            j += 1
        norm = np.linalg.norm(data[i])
        if noise == False:
            data[i] = data[i]/norm
        if noise==True:
            noise_term = (random.uniform(-1,1) * noise_amplitude)
            print(noise_term)
            data[i] = (data[i]/norm) + noise_term
        i += 1
    j = offset_dimension
    if offset != 0:
        i = 0
        while i < n_samples:
            data[i,j] = offset
            i += 1
    data= pd.DataFrame(data)
    # print(data)
    return data

def mst(data):
    # Given a dataset, this function returns a tree structure containing the
    # minimum spanning tree for the dataset
    dist_matrix = scipy.spatial.distance_matrix(data,data)
    tree = scipy.sparse.csgraph.minimum_spanning_tree(dist_matrix)
    return tree

def get_coords(tree):
    # This is a utility function used by the function ged below
    coo = tree.tocoo()
    first = coo.row
    second = coo.col
    coords = []
    i = 0
    while i < len(coo.row):
        coord = tuple((first[i],second[i]))
        coords.append(coord)
        # print(coord)
        i += 1
    return coords

def ged(tree1,tree2):
    # Given two trees, this function finds the graph edit distance between them.
    tree1_coords= get_coords(tree1)
    tree1_inversed = [(item[1],item[0]) for item in tree1_coords]
    tree2_coords= get_coords(tree2)
    tree2_inversed = [(item[1],item[0]) for item in tree2_coords]
    cost = 0
    j = 0
    while j < len(tree1_coords):
        if tree1_coords[j] not in tree2_coords:
            if tree1_inversed[j] not in tree2_coords:
                cost += 1
        j += 1
    j = 0
    while j < len(tree1_coords):
        if tree2_coords[j] not in tree1_coords:
            if tree2_inversed[j] not in tree1_coords:
                cost += 1
        j += 1
    return cost
