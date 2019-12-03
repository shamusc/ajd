
# First, import the hypersphere function:
from calc_ajd import hypersphere

# When you call the function, specify
  # the dimensionality of the hypersphere,
  # the number of samples,
  # and the ambient dimensionality

data = hypersphere(n_dimensions=5,
                  n_samples=1000,
                  k_space=20)

# For this, we'll need the functions ndr and ajd:
from calc_ajd import ndr, ajd
# First, we perform dimensionality reduction:
embedding = ndr(data,method='PCA',dim=2)
# Next, we calculate the Average Jaccard Distance:
distortion = ajd(data, embedding)
# Finally, we print our result
print("AJD of embedding:  ", distortion)

# Declare some lists to store results:
dims = []
results = []
# For each dimension from 1 to 10:
i = 1
while i < 11:
  # Perform dimensionality reduction:
  embedding = ndr(data,method='PCA',dim=i)
  # Calculate GED:
  avg_jaccard_dist = ajd(data,embedding)
  # Add embedding dimensions and AJD result to lists:
  dims.append(i)
  results.append(avg_jaccard_dist)
  i += 1
# Print results:
print(dims)
print(results)
# Remember to import the package!
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 4.5))
plt.plot(dims,results,'bo-')
plt.title('Hypersphere Data \n Dimension of Embedding vs. Avg. Jaccard Distance')
plt.xlabel('Embedding Dimension')
plt.ylabel('Avg. Jaccard Distance')
plt.savefig('sample_result_1.png')

from calc_ajd import ged, mst
# Declare some lists to store results:
dims = []
results = []
# We only need to find the minimum spanning tree in the original data once:
highD_tree = mst(data)
# For each dimension from 1 to 10:
i = 1
while i < 11:
  # Perform dimesnionality reduction:
  embedding = ndr(data,method='PCA',dim=i)
  # Now we find the minimum spanning tree in the new representation of the data:
  lowD_tree = mst(embedding)
  # We calculate the Graph Edit Distance between the trees:
  graph_edit_distance = ged(highD_tree,lowD_tree)
  # And add our data to the relevant lists:
  dims.append(i)
  results.append(graph_edit_distance)
  i += 1
plt.figure(figsize=(8, 4.5))
plt.plot(dims,results,'ro-')
# plt.plot(dims,results,'r--')
plt.title('Practice Data \n Dimension of Embedding vs. Graph Edit Distance btwn. Minimum Spanning Trees')
plt.xlabel('Embedding Dimension')
plt.ylabel('Graph Edit Distance')
plt.savefig('sample_result_2.png')
