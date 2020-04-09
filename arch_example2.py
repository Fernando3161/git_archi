'''
Created on 6 Apr 2020

@author: Besitzer
'''
import pygmo as pg
import time
import warnings
from numba import jit, float64, errors
warnings.filterwarnings("ignore", "", errors.NumbaDeprecationWarning)
warnings.filterwarnings("ignore", "", errors.NumbaWarning)
import numpy as np

class toy_problem_o2:
    
    def __init__(self, dim):
        self.dim = dim
        
    @jit
    def fitness(self, x):
        return [toy_problem_o2._a(x)[0], toy_problem_o2._b(x)[0], 
                -toy_problem_o2._c(x)[0]]
    
    @jit(float64[:](float64[:]), nopython=True)
    def _a(x):
        retval = np.zeros((1,))
        for x_i in x:
            retval[0]+=x_i
        return retval
    
    @jit(float64[:](float64[:]), nopython=True)
    def _b(x):
        retval = np.zeros((1,))
        sqr = np.zeros((1,))
        for x_i in x:
            sqr[0] += x_i*x_i
        retval[0]=1.-sqr[0]
        return retval
    
    @jit(float64[:](float64[:]), nopython=True)
    def _c(x):
        retval = np.zeros((1,))
        for x_i in x:
            retval[0]+=x_i
        return retval
    
    
    def gradient(self, x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)  # numerical gradient

    def get_nec(self):
        return 1

    def get_nic(self):
        return 1

    def get_bounds(self):
        return ([-1] * self.dim, [1] * self.dim)

    def get_name(self):
        return "A toy problem, 2nd optimization"

    def get_extra_info(self):
        return "\tDimensions: " + str(self.dim)

def archipielago_opt(f,n):
    start = time.time()
    funct=f(n)
    name=funct.get_name()

    a_cstrs_sa = pg.algorithm(pg.cstrs_self_adaptive(iters=1000))
    t1=time.time()
    p_toy = pg.problem(funct)
    p_toy.c_tol = [1e-4, 1e-4]
    archi = pg.archipelago(n=16, algo=a_cstrs_sa, prob=p_toy, pop_size=10)
    archi.evolve(2)
    archi.wait_check()

if __name__ == '__main__':
    archipielago_opt(toy_problem_o2,2)
    
    