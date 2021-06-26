# Implementation of LSH and Linear search for Big data CSV file and plotting the differences


import numpy as np
import random
import time
import pdb
import unittest
from PIL import Image
from pyspark import SparkContext, SparkConf
import csv
import matplotlib.pyplot as plt
from numpy import linalg as dt
from pyspark.sql import SQLContext
from scipy.spatial import distance

#Class LSH - read the data into rdd , mapping hash vectors 
#filtering the rdd . finding distances to find the nearest neighbours
class LSH:

    #default constructor with parameters filename, hashfunction length and size of hash vector
    def __init__(self, filename, k, L):
        """
        Initializes the LSH object
        filename - name of file containing dataframe to be searched
        k - number of thresholds in each function
        L - number of functions
        """
        # do not edit this function!
        self.sc = SparkContext()
        self.k = k
        self.L = L
        self.A = self.load_data(filename)
        self.functions = self.create_functions()
        self.hashed_A = self.hash_data()
    
    
    def l1(self, u, v):
        """
        Finds the L1 distance between two vectors
        u and v are 1-dimensional Row objects
        """
        p=np.sum(abs(np.array(u)-np.array(v)))
        return p

    
    # load_data function will take the filename from the lsh object and will split the rows and convert them as strings
    # Those string rows are converetd to integer values and zipped with index and will return RDD
    def load_data(self, filename):
        """
        Loads the data into a spark DataFrame, where each row corresponds to
        an image patch -- this step is sort of slow.
        Each row in the data is an image, and there are 400 columns.
        """
        #sqlcontext = SQLContext(self.sc)
        #df = sqlcontext.read.format('com.databricks.spark.csv').options(header='false', inferschema='true').load(filename)
        #print (df.count())
        df = self.sc.textFile(filename).map(lambda line: line.split(","))
        l = df.map(lambda w: [int(float(c)) for c in w]).zipWithIndex()
        return l
#         raise NotImplementedError


    # This function will create hashvector of form 10101.. for the passed function from create_functions
    def create_function(self, dimensions, thresholds):
        """
        Creates a hash function from a list of dimensions and thresholds.
        """
        def f(v):
            s = ''
            for i in range(len(dimensions)):
                if(float(v[dimensions[i]])>=thresholds[i]):
                    s +='1'
                else:
                    s +='0'
            return s
#             raise NotImplementedError
        return f


    # This function will take L and k values from lSH object and will generate functions list
    # Every time it will generate random values for dimensions and thresholds
    def create_functions(self, num_dimensions=400, min_threshold=0, max_threshold=255):
        """
        Creates the LSH functions (functions that compute L K-bit hash keys).
        Each function selects k dimensions (i.e. column indices of the image matrix)
        at random, and then chooses a random threshold for each dimension, between 0 and
        255.  For any image, if its value on a given dimension is greater than or equal to
        the randomly chosen threshold, we set that bit to 1.  Each hash function returns
        a length-k bit string of the form "0101010001101001...", and the L hash functions 
        will produce L such bit strings for each image.
        """
        functions = []
        for i in range(self.L):
            dimensions = np.random.randint(low = 0, 
                                    high = num_dimensions,
                                    size = self.k)
            thresholds = np.random.randint(low = min_threshold, 
                                    high = max_threshold + 1, 
                                    size = self.k)

            functions.append(self.create_function(dimensions, thresholds))
        return functions

    
    def hash_vector(self,v):
        """
        Hashes an individual vector (i.e. image).  This produces an array with L
        entries, where each entry is a string of k bits.
        """
        # you will need to use self.functions for this method
        x = np.array([f(v[0]) for f in self.functions])
        #print (x)
        return x
#         raise NotImplementedError
    
    
    # Hash_data function will hash the vector for each row and will map to the rdd
    def hash_data(self):
        """
        Hashes the data in A, where each row is a datapoint, using the L
        functions in 'self.functions'
        """
        # you will need to use self.A for this method
        func = self.functions
        # For each row of file, it will create hash vectors and will map that hash function to the RDD
        query = self.A.map(lambda q: q + ([f(q[0]) for f in func],))
        #print(query.take(1))
        return query
        #raise NotImplementedError


        
    # Get_candidates will filter the RDD, if any of hash vector matches with query_index hash vectors
    # If both vectors of one hash function same, it will place in the bucket
    def get_candidates(self,query_index):
        """
        Retrieve all of the points that hash to one of the same buckets 
        as the query point.  Do not do any random sampling (unlike what the first
        part of this problem prescribes).
        Don't retrieve a point if it is the same point as the query point.
        """
        # you will need to use self.hashed_A for this method
        bucket1 = self.hashed_A
        bucket = bucket1.filter(lambda z: (z[2] != query_index[2]) and (any(set(z[2]) & set(query_index[2]))))
        #print(bucket)
        return bucket
#         raise NotImplementedError



    # LSH_search will call the get_candidates and will calculate distances and map them to the RDD
    # From RDD, we will pick distances and indexes to new RDD as tuple
    # RDD will be sorted by distance value and will calculate end time
    # with the provided neighbour value, it will return nearest neighbours and end time.
    def lsh_search(self,query_index, num_neighbors = 10):
        """
        Run the entire LSH algorithm
        """
        def l1(u,v):
            return dt.norm(np.array(u)-np.array(v), ord=1)
        
        start_time = time.time()
        #print(start_time)
        buckets = self.get_candidates(query_index)
        distance1 = buckets.map(lambda p : p + (l1(p[0],query_index[0]),))
        distance_sort = distance1.map(lambda y : (y[3],y[1]))
        distance_sorted = distance_sort.sortByKey()
        lsh_End_time = time.time()- start_time
        return (distance_sorted.take(num_neighbors),lsh_End_time)
#         raise NotImplementedError


# Plots images at the specified rows and saves them each to files.
def plot(A, row_nums, base_filename):
    for row_num in row_nums:
        patch = np.reshape(A[row_num, :], [20, 20])
        im = Image.fromarray(patch)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(base_filename + "-" + str(row_num) + ".png")

        
# Finds the nearest neighbors to a given vector, using linear search.
def linear_search(A, query_index, num_neighbors):
    close_neighbors = {}
    
    def l1(u,v):
        return dt.norm(u-v, ord=1)
    linear_start_time = time.time()
    for i in range(len(A)):
        dis = l1(A[i],query_index)
        if (dis == 0):
            continue
        else:
            close_neighbors[i] = dis
    
    linear_end_time = time.time() - linear_start_time
    return (sorted(close_neighbors.items(), key = lambda j:(j[1],j[0]))[:num_neighbors] ,linear_end_time)
#     raise NotImplementedError


# lsh_error will calculate the error between linearsearch and LSH distance
# Write a function that computes the error measure
def lsh_error(LSH_Distance, Linear_Distance):
    Error_value = 0.0
    Lsh_distance_sum = sum(LSH_Distance)
    Linear_distance_sum = sum(Linear_Distance)
    Error_value = Lsh_distance_sum/Linear_distance_sum
    return Error_value
#     raise NotImplementedError

#### TESTS #####

class TestLSH(unittest.TestCase):
    def test_l1(self):
        u = np.array([1, 2, 3, 4])
        v = np.array([2, 3, 2, 3])
        self.assertEqual(l1(u, v), 4)

    def test_hash_data(self):
        f1 = lambda v: sum(v)
        f2 = lambda v: sum([x * x for x in v])
        A = np.array([[1, 2, 3], [4, 5, 6]])
        self.assertEqual(f1(A[0,:]), 6)
        self.assertEqual(f2(A[0,:]), 14)

        functions = [f1, f2]
        self.assertTrue(np.array_equal(hash_vector(functions, A[0, :]), np.array([6, 14])))
        self.assertTrue(np.array_equal(hash_data(functions, A), np.array([[6, 14], [15, 77]])))


if __name__ == '__main__':
#    unittest.main() ### TODO: Uncomment this to run tests
    # create an LSH object using lsh = LSH(k=16, L=10)
    """
    Your code here
    """
    
    linearsearchtime=[]
    lshsearchtime=[]
    lsh=LSH("patches.csv",24,10)
    A=pd.read_csv("patches.csv",header=None)
    
    lsh_srh=[]
    for i in range(100,1001,100):
        start2=timeit.default_timer()
        lsh_srh.append(lsh.lsh_search(i,3))
        print(lsh_srh)
        stop2=timeit.default_timer()
        lshsearchtime.append(stop2-start2)
    avgt2=sum(lshsearchtime)/len(lshsearchtime)
    print(avgt2)
    print("lsh")
    print(lsh_srh)
    
    linsrh=[]
    for i in range(100,1001,100):
        start=timeit.default_timer()
        linsrh.append(linear_search(A,i,3))
        stop=timeit.default_timer()
        linearsearchtime.append(stop-start)
    print("lin")
    print(linsrh)
    avgtime_linear=sum(linearsearchtime)/len(linearsearchtime)
    print(avgtime_linear)
    
    errorvals=lsh_error(linsrh,lsh_srh)
    print(errorvals)
    
    B = np.genfromtxt ('patches.csv', delimiter=",")
    p1=[]
    p2=[]
    for i in range(len(linsrh[0])):
        p1.append(linsrh[0][i][0])
    
    for j in range(len(lsh_srh[0])):
        p2.append(lsh_srh[0][j][1])
    
    #plotting 10 nearesr neighbors for query index 100
    plot(B, p1, "linear")
    plot(B, p2, "lsh")
    
    
    k = 24
    L = [10,12,14,16,18,20]
    l_error = []
    for i in range(len(L)):
        lsh=LSH("patches.csv",k=k,L=L[i])
        A=pd.read_csv("patches.csv",header=None)
    
        lsh_srh=[]
        for i in range(100,1001,100):
            start2=timeit.default_timer()
            lsh_srh.append(lsh.lsh_search(i,3))
            print(lsh_srh)
            stop2=timeit.default_timer()
            lshsearchtime.append(stop2-start2)
        avgt2=sum(lshsearchtime)/len(lshsearchtime)
        print(avgt2)
        print("lsh")
        print(lsh_srh)
        
        linsrh=[]
        for i in range(100,1001,100):
            start=timeit.default_timer()
            linsrh.append(linear_search(A,i,3))
            stop=timeit.default_timer()
            linearsearchtime.append(stop-start)
        print("lin")
        print(linsrh)
        avgtime_linear=sum(linearsearchtime)/len(linearsearchtime)
        print(avgtime_linear)

        errorvals=lsh_error(linsrh,lsh_srh)
        l_error.append(errorvals)
        
    L = 10
    k = [16,18,20,22,24]
    k_error = []
    for i in range(len(k)):
    
        lsh=LSH("patches.csv",k=k[i],L=L)
        A=pd.read_csv("patches.csv",header=None)

        lsh_srh=[]
        for i in range(100,1001,100):
            start2=timeit.default_timer()
            lsh_srh.append(lsh.lsh_search(i,3))
            print(lsh_srh)
            stop2=timeit.default_timer()
            lshsearchtime.append(stop2-start2)
        avgt2=sum(lshsearchtime)/len(lshsearchtime)
        print(avgt2)
        print("lsh")
        print(lsh_srh)
        
        linsrh=[]
        for i in range(100,1001,100):
            start=timeit.default_timer()
            linsrh.append(linear_search(A,i,3))
            stop=timeit.default_timer()
            linearsearchtime.append(stop-start)
        print("lin")
        print(linsrh)
        avgtime_linear=sum(linearsearchtime)/len(linearsearchtime)
        print(avgtime_linear)

        errorvals=lsh_error(linsrh,lsh_srh)
        k_error.append(errorvals)
        
    #plotting k vs error
    plt.plot([16,18,20,22,24],k_error)
    plt.xlabel("function of K")
    plt.ylabel("Error")
    plt.title("K vs Error")
    plt.show()
    
    
    #plotting L vs error
    plt.plot([10,12,14,16,18,20],l_error)
    plt.xlabel("function of L")
    plt.ylabel("Error")
    plt.title("L vs Error")
    plt.show()
    
    
