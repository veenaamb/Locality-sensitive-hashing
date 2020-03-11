# Implementation of LSH and Linear search for Big data CSV file and plotting the differences

#importing numpy,random,time,csv,matlab
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
#from pyspark.sql import SQLContext
#code change
#from scipy.spatial import distance

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
        #self.sc = SparkContext()
        conf = SparkConf()
        self.sc = SparkContext().getOrCreate(conf = conf)
        self.k = k
        self.L = L
        #get an RDD splitting thr rows and zipping with index values
        self.A = self.load_data(filename)
        #get functions list to call create_function which will generate hash vectors
        self.functions = self.create_functions()
        #Generate bucket which filter RDD if one hash vector of image same with the query_index
        self.hashed_A = self.hash_data()
    
    def __del__(self):
        #To stop the current sparksession
        self.sc.stop()
    # I have encountered pickling issue which mentions that i am attempting to refernce sparkcontext  
    # from a broadcast variable, action or transformation - SPARK 5063
    # To avoid the above error, i have used l1 function inside the lsh_search
    # TODO: Implement this
    '''def l1(self, u, v):
        """
        Finds the L1 distance between two vectors
        u and v are 1-dimensional Row objects
        """
        d = dt.norm(np.array(u)-np.array(v))
        return d
        raise NotImplementedError'''

    # load_data function will take the filename from the lsh object and will split the rows and convert them as strings
    # Those string rows are converetd to integer values and zipped with index and will return RDD
    # TODO: Implement this
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
        raise NotImplementedError

    # This function will create hashvector of form 10101.. for the passed function from create_functions
    # TODO: Implement this
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
            raise NotImplementedError
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

    # This function is implemented but never used
    # TODO: Implement this
    def hash_vector(self,v):
        """
        Hashes an individual vector (i.e. image).  This produces an array with L
        entries, where each entry is a string of k bits.
        """
        # you will need to use self.functions for this method
        x = np.array([f(v[0]) for f in self.functions])
        #print (x)
        return x
        raise NotImplementedError
    
    # Hash_data function will hash the vector for each row and will map to the rdd
    # TODO: Implement this
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
        raise NotImplementedError

    # Get_candidates will filter the RDD, if any of hash vector matches with query_index hash vectors
    # If both vectors of one hash function same, it will place in the bucket
    # TODO: Implement this
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
        raise NotImplementedError

    # LSH_search will call the get_candidates and will calculate distances and map them to the RDD
    # From RDD, we will pick distances and indexes to new RDD as tuple
    # RDD will be sorted by distance value and will calculate end time
    # with the provided neighbour value, it will return nearest neighbours and end time.
    # TODO: Implement this
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
        raise NotImplementedError

# Plots images at the specified rows and saves them each to files.
def plot(A, row_nums, base_filename):
    for row_num in row_nums:
        patch = np.reshape(A[row_num, :], [20, 20])
        im = Image.fromarray(patch)
        if im.mode != 'RGB':
            im = im.convert('RGB')
        im.save(base_filename + "-" + str(row_num) + ".png")

# Finds the nearest neighbors to a given vector, using linear search.
# TODO: Implement this
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
    raise NotImplementedError

# lsh_error will calculate the error between linearsearch and LSH distance
# Write a function that computes the error measure
# TODO: Implement this
def lsh_error(LSH_Distance, Linear_Distance):
    Error_value = 0.0
    Lsh_distance_sum = sum(LSH_Distance)
    Linear_distance_sum = sum(Linear_Distance)
    Error_value = Lsh_distance_sum/Linear_distance_sum
    return Error_value
    raise NotImplementedError

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

    ### TODO: Write your tests here (they won't be graded, 
    ### but you may find them helpful)


# calling the main function
if __name__ == '__main__':
#    unittest.main() ### TODO: Uncomment this to run tests
    # create an LSH object using lsh = LSH(k=16, L=10)
    """
    Your code here
    """
    # Opening patches.csv file and read into numpy array
    with open('patches.csv','r') as f:
        A = list(csv.reader(f, delimiter= ","))
    A = np.array(A[:],dtype = np.float)
    
    # calling the LSH class object with values k=24 and L=10
    lsh = LSH('patches.csv',k=24,L=10)
    # below will retrive the nearest neighbours and time for lsh search
    dist_index,te = lsh.lsh_search(lsh.hashed_A.collect()[99],10)
    #print("The value of L and k are: ["+str(self.L)+","+str(k)+"]")
    # printing the ten nearest neighbours
    print("The ten nearest neighbours with LSH search for given image are: ")
    print(dist_index)
    #printing the time taken for LSH search
    print("Time taken for LSH_Search is: "+ str(te))
    
    #plotting the images of nearest_neighbours in LSH search
    row_lsh = []
    lsh_distances = []
    for i in range(len(dist_index)):
        row_lsh.append(dist_index[i][1])
        lsh_distances.append(dist_index[i][0])
    plot(A,row_lsh,'image_LSH_Search')
   
    # for plotting nearest neighbours in linear search
    row_linear = []
    linear_distances =[]
    # calling the linear search function to retrieve nearest neighbours and time taken for it.
    linear_d,linear_t = linear_search(A, A[99], 10)
    #printing the top ten nearest neighbours from linear search
    print("The ten nearest neighbours with Linear search for given image are: ")
    print(linear_d)
    #printing the time taken for linear search
    print("Time taken for Linear_Search is: "+ str(linear_t))
    #plotting the images of nearest_neighbours in linear search
    for i in range(len(dist_index)):
        row_linear.append(linear_d[i][0])
        linear_distances.append(linear_d[i][1])
    plot(A,row_linear,'image_linear_Search')
   # print(linear_distances)
    #print(row_linear)
    #calculating the error between linear search and lSH search
    error_v = lsh_error(lsh_distances,linear_distances)
    print("The Error value is: "+ str(error_v))
    
    # plotting the original image
    plot(A,[99],'Original_image')
    
    # it will stop the current sparksesion
    lsh.sc.stop()
    
    # I tried running for all values of L and K by stopping the sparkcontext and re-run
    # with new L , K values but it throws error as Python worker failed to connect back
    # It's because it is trying to reconnect sparkcontext bnut it is failing
    # I found some information on stackoverflow that it fails to connect again in the latest versions of spark
    # Hence, commented the code below
    #To run the below code in previous versions, please comment the above code and uncomment below
    
    """
    #looping through L and getting the change in errors
    error_l = []
    for L in range(10,22,2):
        with open('patches.csv','r') as f:
            A = list(csv.reader(f, delimiter= ","))  
        A = np.array(A[:],dtype = np.float)
        lsh_distance = []
        linear_dis = []
        lsh_time = []
        linear_time = []
        lsh = LSH('patches.csv', k=24, L=L)
        for i in range(99,1000,100):
            distance = []
            f,t = lsh.lsh_search(lsh.hashed_A.collect()[i], 3)
            for i in range(len(f)):
                distance.append(f[i][0])
            lsh_distance.append(sum(distance))
            lsh_time.append(t)
        
            lid = []
            ld,lt = linear_search(A, A[i], 3)
            for i in range(len(ld)):
                lid.append(ld[i][1])
            linear_dis.append(sum(lid))
            linear_time.append(lt)
            
        #print("Average Time for LSH search: ", sum(lsh_time)/10)   
        error_l.append(lsh_error(lsh_distance, linear_dis))    
        lsh.sc.stop()# stoping the sparkcontext
        #error_l.append(lsh_error(lsh_distance, linear_dis))
    
    #looping through k and getting the change in errors 
    error_k = []
    for k in range(16,26,2):
        with open('patches.csv','r') as f:
            A = list(csv.reader(f, delimiter= ","))
        A = np.array(A[:],dtype = np.float)
        lsh_distance = []
        linear_dis = []
        lsh_time = []
        linear_time = []
        lsh = LSH('patches.csv', k=k , L=10)
        for i in range(99,1000,100):
            distance = []
            f,t = lsh.lsh_search(lsh.hashed_A.collect()[i], 3)
            for i in range(len(f)):
                distance.append(f[i][0])
            lsh_distance.append(sum(distance))
            lsh_time.append(t)
        
            lid = []
            ld,lt = linear_search(A, A[i], 3)
            for i in range(len(ld)):
                lid.append(ld[i][1])
            linear_dis.append(sum(lid))
            linear_time.append(lt)
            
        #print("Average Time for Linear search: ", sum(linear_time)/10)
        error_k.append(lsh_error(lsh_distance, linear_dis))
        lsh.sc.stop()   
        #error_k.append(lsh_error(lsh_distance, linear_dis))
        
    #printed the average time in different cell when ran in jupyter
        
    print("Average Time for LSH search: ", sum(lsh_time)/10)
    print("Average Time for Linear search: ", sum(linear_time)/10)
        
    #  print("Error is :",lsh_error(lsh_distance, linear_dis))
    

    #plotting the L vs Error graph
    plt.plot([10,12,14,16,18,20], error_l)
    plt.xlabel('function of L')
    plt.ylabel('Error Value')
    plt.title('L vs Error Value')
    plt.show()


    #plotting the k vs Error graph
    plt.plot([16,18,20,22,24], error_k)
    plt.xlabel('function of K')
    plt.ylabel('Error Value')
    plt.title('K vs Error Value')
    plt.show()


    with open('patches.csv','r') as f:
            A = list(csv.reader(f, delimiter= ","))
    A = np.array(A[:],dtype = np.float)
    plot(A,[99,100],'img')# plotting the 100th label
    """