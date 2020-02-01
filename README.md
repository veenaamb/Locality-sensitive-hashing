# Locality-sensitive-hashing
Application of LSH to the problem of finding approximate near neighbors

Locality-sensitive hashing (LSH) is an approximate nearest neighbor search and clustering method for high dimensional data points.
Locality-Sensitive functions take two data points and decide about whether or not 
they should be a candidate pair. LSH hashes input data points multiple times in a way that similar data points map to 
the same "buckets" with a high probability than dissimilar data points. 
The data points map to the same buckets are considered as candidate pair.

When Normal hashes are meant to avoid collisions, Locality-sensitive hashes are designed to cause collisions. 
The more similar the input data is, the more similar the resulting hashes will be, with a small and predictable error rate.

This makes them a very useful tool for large scale data mining, as a component in:

* duplicate detection
* fuzzy-matching database records or similar documents
* nearest-neighbours clustering and classification
* content recommendation
