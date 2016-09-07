
import sys
import matlab
import numpy as np
from scipy.sparse.linalg.eigen.arpack import eigsh as largest_eigsh
from  numpy.linalg import inv
eng = None


def convert_type(arr):
    if isinstance(arr, np.ndarray):
        arr = arr.tolist()

    if isinstance(arr, list):
        arr = matlab.double(arr)

    return arr


def get_eng():
    global eng    
    if eng == None:
        import matlab.engine
        eng = matlab.engine.start_matlab()
    return eng


def sample_dpp(L,k=0):
    '''
    Wrapper function for the sample_dpp Matlab code written by Alex Kulesza
    Given a kernel matrix L, returns a sample from a k-DPP.
    L:     kernel matrix: list of lists, 2D numpy array, or matlab.doubles 
    k:     size of the sample from the DPP: int
    return: 1D numpy ndarry of type int for 0-based indicies of the samples
    '''
    # Matlab link setup
    eng = get_eng()
    L = convert_type(L)

    M = eng.decompose_kernel(L)
    if k:
        k = matlab.int64([k])
        S = eng.sample_dpp(M, k)
    else:
        S = eng.sample_dpp(M)

    # matlab returns 1 based indexing, so convert to python indexing
    # Also condition on the return type of sample_dpp()
    if isinstance(S, float):
        S = np.array([S], np.int) - 1
    else:
        # matlab.double object
        S = np.array(S._data, np.int)
        S = S - 1
    return S


# TODO: not sure if this is correct.  It adds 1 to set0 indices, but the construction of the
# L_compset takes place in pure Python.
def sample_conditional_dpp(L,set0,k=None):
    '''
    Wrapper function for the sample_dpp Matlab code written by Alex Kulesza
    Given a kernel matrix L, returns a sample from a k-DPP.
    The code is hacked in a way that if a set A is provied, samples from a conditional 
    dpp given A are produced
    L:     kernel matrix, numpy 2D array
    set0:  index of the conditional elements. Integer numpy array containing the locations 
            (starting in zero) relative to the rows of L.
    k:     size of the sample from the DPP
    '''
    set0 = np.array(set0) + 1  # matlab starts counting in one

    # Calculate the kernel for the marginal
    Id = np.array([1] * L.shape[0])
    Id[set0] = 0
    Id = np.diag(Id)    
    L_compset_full = inv(Id + L)
    L_minor = inv(
        np.delete(
            np.delete(
                L_compset_full, 
                tuple(set0), 
                axis=1),
            tuple(set0),
            axis=0)
        )
    L_compset = L_minor - np.diag([1] * L_minor.shape[0])
    
    # Compute the sample
    sample = sample_dpp(L_compset,k)
    return np.concatenate((set0-1,sample), axis=0)  # back to python indexing


#def sample_dual_conditional_dpp(L,set0,q,k=None):
#    '''
#    Wrapper function for the sample_dpp Matlab code written by Alex Kulesza
#    Given a kernel matrix L, returns a sample from a dual k-DPP.
#    The code is hacked in a way that if a set0 A is provied, samples from a conditional 
#    dpp given A are produced
#    L:     kernel matrix
#    set0:   index of the conditional elements. Integer numpy array containing the locations 
#           (starting in zero) relative to the rows of L.
#    q:     is the number of used eigenvalues
#    k:     size of the sample from the DPP
#    '''
#    # Calculate the kernel of the marginal
#    Id = np.array([1]*L.shape[0])
#    Id[set0] = 0
#    Id = np.diag(Id)    
#    L_compset_full = inv(Id + L)
#    L_minor = inv(np.delete(np.delete(L_compset_full,tuple(set0), axis=1),tuple(set0),axis=0))
#    L_compset = L_minor - np.diag([1]*L_minor.shape[0]) 
#    
#    # Take approximated sample
#    sample = sample_dual_dpp(L_compset,q,k-1)
#    if k==2: sample = [sample]
#    return np.concatenate((set0,sample) ,axis=0)
#
#
#
#def sample_dual_dpp(L,q,k=None):
#    '''
#    Wrapper function for the sample_dual_dpp Matlab code written by Alex Kulesza
#    Given a kernel matrix L, returns a sample from a k-DPP.
#    
#    L is the kernel matrix
#    q is the number of used eigenvalues
#    k is the number of elements in the sample from the DPP
#    '''
#    # Matlab link
#    global eng    
#    if eng == None:
#        import matlab.engine
#        eng = matlab.engine.start_matlab()
#        
#    # Extract the feature matrix from the kernel
#    evals, evecs = largest_eigsh(L,q,which='LM')
#    B = np.dot(evecs,np.diag(evals))
#    
#    # load values in Matlab and get sample
#    eng.put('B',B)
#    
#    if k!=None: 
#        k = np.array([[k]])  # matlab only undernstand matrices 
#        eng.put('k',k)
#        eng.eval("dpp_sample = sample_dual_dpp(B,decompose_kernel(B'*B),k)")
#    else:
#        eng.eval("dpp_sample = sample_dual_dpp(B,decompose_kernel(B'*B))")
#        
#    dpp_sample = eng.get('dpp_sample')
#    return dpp_sample.astype(int)










