#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 26 11:22:44 2022

@author: Tighe_Clough
"""

import pandas as pd
import numpy as np
import copy
import time

# can inherit gridsearchcv and use for this on different warping windows?


class knnDtwAdap:
    """ 
    K-nearest neighbor classifier using dynamic time warp (DTW) as the
    distance measure. Expands use of DTW to multi-dimensional time series (MDT).
    Employs both independent and dependent DTW measures.
    
    Parameters
    ----------
    n_neighbors: int, optional (default = 1)
        Number of neighbors to use for KNN 
        
    max_warping_window: int, optional (Default = infinity)
    
    """
    
    def __init__(self, n_neighbors: int = 1, max_warping_window: int = np.inf, squared: bool = False):
        self.n_neighbors = n_neighbors
        self.max_warping_window = max_warping_window
        self.squared = squared
        
    def fit(self, X, Y):
        """ 
        Fit the model using X as training data and Y as class labels 
        
        Parameters
        ---------
        X: pandas dataframe of shape [n_steps*n_sequence, n_sensors+2]
            Training data set for input into KNN classifier
            Long data format:
                Column1: Sequence number (sequence is one stage of recording)
                    often corresponds to one action
                Column2: Step (usually timepoints)
                Column3-n:Sensormetrics data, each column one sensor or metrics
            
        Y: pandas dataframe of array of shape [n_sequences, 2]
            one-to-many relationship with x
            lists 
                Column1: sequence
                Column2: corresponding activity label for seeqwuence
        
        squared: bool, optional (default = False)
            False: use Euclidean distance (absolute value)
            True: use square Euclidean distance
        """
        
        self.X = X
        self.Y = Y
        
        # threshold through decision tree
        self.threshold = self.learn_threshold()    
    
    def dtw_mat(self, dist_mat):
        """ Finds DTW distance between two series 
        
        Parameters
        ----
        dist_matrix: distance matrix (np array) to calculate dtw

        Returns
        -------
        DTW distance of distance matrix given warping
        """
        M, N = dist_mat.shape
        
        # Initialize cost matrix
        cost_mat = np.full((M, N), np.inf)
        
        cost_mat[0,0] = dist_mat[0,0]
        
        # first layer follows the same patters
        for j in range(1,self.max_warping_window+1):
            cost_mat[0,j] = cost_mat[0,j-1] + dist_mat[0,j]
        
        for i in range(1,self.max_warping_window+1):
            cost_mat[i,0] = cost_mat[i-1,0] + dist_mat[i,0]
        
        # populate rest of matrix
        for i in range(1, M):
            for j in range(max(1, i - self.max_warping_window),
                           min(N, i + self.max_warping_window)):
                adjacent = [cost_mat[i, j-1], cost_mat[i-1,j-1], cost_mat[i-1,j]]
                # calculate matrix total sum
                cost_mat[i,j] = min(adjacent) + dist_mat[i,j]
        
        ending_val = cost_mat[-1,-1]
        
        if self.squared == True:
            dtw_cost = ending_val**.5
        else:
            dtw_cost = ending_val
        
        return dtw_cost
    
    def dtw_i(self, seq1, seq2):
        """ Finds independent DTW (DTWi) distance between two samples of MDT
        sensors/series must correspond to same columns in each sensor
        
        Parameters
        ----------
        sample1: array of shape [n_steps, n_sensors/series]
        
        sample2: array of shape [n_steps, n_sensors/series]
        
        Returns
        -------
        DTWi distance between seq1 and seq2
        """
        
        # some nice numpy to get arrays
        M, N = seq1.shape
        seq1_d = copy.deepcopy(seq1.T.reshape((N, M,1)))
        
        M2, N2 = seq2.shape
        seq2_d = copy.deepcopy(seq2.T.reshape((N2, 1,M2)))
        
        base_dist = seq1_d - seq2_d
        
        if self.squared == True:
            fin_dist = base_dist**2
        else:
            fin_dist = np.absolute(base_dist)
        
        # calculate cost from each matrix
        dtw_i_cost = 0
        for cost_matrix in fin_dist:
            dtw_i_cost += self.dtw_mat(cost_matrix)
        
        return dtw_i_cost
        
    
    def dtw_d(self, seq1, seq2):
        """ Finds dependent DTW (DTWd) distance between two samples of MDT
        sensors/series/dimesnions must correspond to same columns in each sensor
            
        Parameters
        ----------
        seq1: array of shape [n_steps, n_sensors/series]
            sequence where each column correspondes to a dimension/sensor
        
        seq2: array of shape [n_steps, n_sensors/series]
            sequence where each column correspondes to a dimension/sensor
        
        Returns
        -------
        DTWd distance between seq1 and seq2
        """    
        
        # some nice numpy to get arrays
        M, N = seq1.shape
        seq1_d = copy.deepcopy(seq1.T.reshape((N,M,1)))
        
        M2, N2 = seq2.shape
        seq2_d = copy.deepcopy(seq2.T.reshape((N2,1,M2)))
        
        base_dist = seq1_d - seq2_d
        
        if self.squared == True:
            fin_dist = np.sum(base_dist**2, axis=0)
        else:
            fin_dist = np.sum(np.absolute(base_dist), axis=0)
        
        dtw_d_cost = self.dtw_mat(fin_dist)
        
        return dtw_d_cost
        
    def learn_threshold(self):
        """ Calculates threshold to for adaptive DTW. 
        Threshold determines when to use DTWi or DTWd
        
        Returns
        -------
        DTW adaptive threshold
        """    
        s_isuccess, s_dsuccess = self.find_scores()
        
        # compare set emptiness to determine threshold score
        if not s_isuccess and not s_dsuccess: # both empty, use 1
            threshold = 1
        
        elif s_isuccess and not s_dsuccess: # independent win over
            threshold = min(s_isuccess)
        
        elif not s_isuccess and s_dsuccess: # dependent win over
            threshold = max(s_dsuccess)
        
        elif not s_isuccess and not s_dsuccess: # bott not empty "Decision tree"
            threshold = self.decider(lowers=s_isuccess, highers=s_dsuccess)
        
        return threshold
    
    def decider(self, lowers, highers):
        """ Finds best cutoff point, staying above most elements in lowers and
        least elements in highers. Actually using logistic regression for this,
        classifying highers as 1 and lowers as 0.
        
        Parameters
        ----------
        lowers: set
            low values to stay above
            
        highers: set
            high values to stay under
            
        Returns
        -------
        DTW adaptive threshold
        """
        
        # label lowers as 0 and highers as 1
        X = np.array(list(lowers)+list(highers)).reshape((-1,1))
        
        zeros = np.zeros((len(lowers),1))
        ones = np.ones((len(highers),1))
        
        y = np.append(zeros, ones, axis=0)
        
        # two thetas
        t0,t1 = self.log_reg(X, y)
        
        # calculate decision boundary
        threshold = -t1 / t0
        
        return threshold
    
    def log_reg(self, X, y, a=0.001, it=10000):
        """ Performs logistic regression and finds optimal thetas
        
        Parameters
        ----------
        X: array
            records
            
        Y: array
            labels
            
        a: int
            alpha learning rate
        
        it: int
            amount of iterations to take
            
        Returns
        -------
        theta: array
            Optimal thetas
        """
        
        # add 1 to the X
        m, n = X.shape
        X1 = np.insert(X, 0, np.ones((1,m)), axis=1)
        
        # initialize fitting parameters
        #theta = np.zeros((n+1,1))
        theta = np.array([-24,.2,.2]).reshape((3,1))
        
        # iterate 
        for i in range(it):
            
            # calculate current gradient
            J, grad = self.cost_and_grad(theta, X1, y)
            # update values
            theta = theta - a * grad
        
        return theta
    
    def cost_and_grad(self, theta, X, y):
        
        # number of training examples
        m = len(y)
        
        pred = self.sigmoid(X @ theta)
        
        grad = (1/m) * (X.T @ (pred - y))
        
        J = -(1/m) * np.sum((y.T @ np.log(pred)) + ((1-y).T @ np.log(1 - pred)))
        
        return J, grad
    
    def sigmoid(self, matrix):
        """ Finds sigmoid values for given matrix
        
        Parameters
        ---------
        matrix: array
            matrix of values
            
        Returns
        -------
        Sigmoid values
        """
        sig_mat = 1 / (1 + np.exp(-matrix))
        
        return sig_mat
        
    def find_scores(self):
        """ For each training record, find one label using DTWi and one using DTWd.
        Compare resulting labels to actual labels.
        If one method (DTWi or DTWd) yield a correct result while the other is incorrect, 
        add its minD/minI (distances to nearest neighbor in each case) ratio to respective list
        
        Returns
        -------
        s_isuccess: set
            threshold when independent yield the correct label and dependent does not
            
        s_dsuccess: set
            threshold when dependent yield the correct label and independent does not
        
        """
        
        # for each sequence in training data, pull out the correct sensor data
        unique_seq = self.X.iloc[:,0].unique()
        ##### sort by just in case
        
        # make array tunnel
        array_tunnel = self.array_tunnel()
        
        # labels comparison dataframe, index to make faster
        label_df = self.Y.set_index(self.Y.columns[0])
        
        # store base_num comp_num answers in np.array
        dis_lookup = np.zeros((2,len(unique_seq),len(unique_seq)))
        
        # initiate sets
        s_isuccess = set()
        s_dsuccess = set()
        
        for base_num in unique_seq:
            # filter to specific sequence
            base_array = array_tunnel[base_num]
            
            # set minimum distances for both independent and dependent methods
            min_distance_i = np.inf
            min_distance_d = np.inf
            
            # set base minimum sequences
            min_sequence_i = base_num
            min_sequence_d = base_num
            
            for comp_num in unique_seq:
                # skip is the same sequence number
                if base_num != comp_num: 
                    if base_num > comp_num: # lookup dis
                        dtw_i, dtw_d = dis_lookup[:,base_num,comp_num]
                    else: # compute new distance mearues
                        # reshape into a matrix
                        comp_array = array_tunnel[comp_num]
                        
                        # find DTWi and DTWd distances
                        dtw_i = self.dtw_i(base_array, comp_array)
                        dtw_d = self.dtw_d(base_array, comp_array)
                        # insert into distance lookup
                        dis_lookup[:,comp_num,base_num] = dtw_i,dtw_d
                        
                    ## might be able to vectorize in the future
                    if dtw_i < min_distance_i:
                        min_sequence_i = comp_num
                        min_distance_i = dtw_i
                        
                    if dtw_d < min_distance_d:
                        min_sequence_d = comp_num
                        min_distance_d = dtw_d
                    
                    
            # compare base label to DTWi and DTWd labels
            base_label = label_df.iloc[base_num,0]
            comp_label_i = label_df[min_sequence_i,0]
            comp_label_d = label_df[min_sequence_d,0]
            
            # i success, add ratio
            if base_label == comp_label_i != comp_label_d:
                s_isuccess.add(min_sequence_d/(min_sequence_i+1))
                
            # d succcess, add ratio
            if base_label == comp_label_d != comp_label_i:
                s_dsuccess.add(min_sequence_d/(min_sequence_i+1))
                
        return s_isuccess, s_dsuccess                  
    
    def test_label(self, X_test):
        """
        
        Parameters
        ----------
        X_test : dataframe, size of [n_test_steps, n_dimensions]
            DESCRIPTION.

        Returns
        -------
        y_test : array, size of [n_test, 2(sequence and label)]
            array of test sequences and labels

        """
        
        self.X_test = X_test
        
        # create tunnel
        train_tunnel = self.array_tunnel(self.X)
        test_tunnel = self.array_tunnel(self.X_test)
        
        # create list of unique sequence in each
        unique_seq_train = self.X.iloc[:,0].unique()
        unique_seq_test = self.X_test.iloc[:,0].unique()
        
        # create distance lookup to save time
        dis_lookup = np.zeros((2,len(unique_seq_test),len(unique_seq_train)))
        
        # labels comparison dataframe, indexed to make faster
        label_df = self.Y.set_index(self.Y.columns[0])
        
        # create array to store labels in 
        y_test_list = list()
        
        # loop through compare to all
        for test_num in unique_seq_test:
            # filter to specific sequence
            test_array = test_tunnel[test_num]
            
            # set minimum distances for both independent and dependent methods
            min_distance_i = np.inf
            min_distance_d = np.inf
            
            # set base minimum sequences
            min_sequence_i = test_num
            min_sequence_d = test_num
            
            for train_num in unique_seq_train:
                if  test_num > train_num: # lookup dis
                    dtw_i, dtw_d = dis_lookup[:,test_num,train_num]
                else: # compute new distance mearues
                    # reshape into a matrix
                    train_array = train_tunnel[train_num]
                    
                    # find DTWi and DTWd distances
                    dtw_i = self.dtw_i(test_array, train_array)
                    dtw_d = self.dtw_d(test_array, train_array)
                    # insert into distance lookup
                    dis_lookup[:,train_num,test_num] = dtw_i,dtw_d
                        
                ## might be able to vectorize in the future
                if dtw_i < min_distance_i:
                    min_sequence_i = train_num
                    min_distance_i = dtw_i
                    
                if dtw_d < min_distance_d:
                    min_sequence_d = train_num
                    min_distance_d = dtw_d
            
            # find current sequence to see a 
            cur_score = min_distance_d/(min_distance_i+1)
            
            if cur_score >= self.threshold: # use independent label
                label = label_df.iloc[min_sequence_i,0]
            else: # use dependent label
                label = label_df.iloc[min_sequence_d,0]
            
            y_test_list.append([test_num, label])
        
        y_test = np.array(y_test_list)
        
        return y_test
    
    def array_tunnel(self, records):
        """
        creates array tunnel, each sequence stacked one behind another

        Returns
        -------
        tunnel : array of dimensions [n_sequences, n_steps_per_series, n_series]
            like file cabinet, where each file is a sequence

        """
        unique_seq_num = len(records.iloc[:,0].unique())
        
        array_tunnel_df = records.iloc[:,2:]
        array_tower = np.array(array_tunnel_df)
        m, n = array_tower.shape
        tunnel = array_tower.reshape((unique_seq_num,-1,n))
        
        return tunnel
    
    def score(self):
        """ retrieve score """
        # read data into tables
        pass

# store certain distances, check if base_num>comp_num (cut time by half)

# only calculate the 0ths row and column you need given window