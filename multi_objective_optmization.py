'''
Created on 15 Apr 2020
This example creates a problem based ont the Fonseca-Fleming Test Function
It provides an optimization through the @jit decorator to convert the problem
to C++
A function is created to observe the influence of the parameters in obtention
of the Pareto Front for the given test function. 

@author: Fernando Penaherrera/UOL/OFFIS
'''
import numpy as np
import pygmo as pg
import time
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit, float64

pg.set_serialization_backend("pickle")

class fonseca_fleming:
    #https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_multi-objective_optimization
    def __init__(self,dim):
        self.dim = dim
    
    def fitness(self, x):
        #Formulation of the fitness function. Minimization problem-> Positive Values
        sums=0
        diff=0
        for i in range(len(x)-1):
            sums+=(x[i]+1/np.sqrt(i+1))**2
            diff+=(x[i]-1/np.sqrt(i+1))**2
        f1 = 1-np.exp(-diff)
        f2 = 1-np.exp(-sums)
        return [f1, f2]
    
    def gradient(self,x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)
    
    def get_nobj(self):
        #Number of Objectives
        return 2
    
    def get_bounds(self):
        #Bounds
        return ([-4] * self.dim,[4]*self.dim)
      
    def get_name(self):
        return "Fonseca-Flaming Test Function"
    
    def get_extra_info(self):
        return "Dimensions " + str(self.dim)

class fonseca_fleming_c:
    #Optimized function to work with @jit decorator
    #https://en.wikipedia.org/wiki/Test_functions_for_optimization#Test_functions_for_multi-objective_optimization
    def __init__(self,dim):
        self.dim = dim

    def fitness(self, x):
        return [fonseca_fleming_c._a(x)[0],
                fonseca_fleming_c._b(x)[0]]

 
    @jit(float64[:](float64[:]),nopython=True)
    def _a(x):
        a = np.zeros((1,))
        diff=np.zeros((1,))
        for i in range(len(x)-1):
            k=1./np.sqrt(i+1)
            diff[0]+=(x[i]-k)*(x[i]-k)
        a[0] = 1.-np.exp(-diff[0])
        return a
    
    @jit(float64[:](float64[:]),nopython=True)
    def _b(x):
        b = np.zeros((1,))
        sums =np.zeros((1,))
        for i in range(len(x)-1):
            k=1./np.sqrt(i+1)
            sums[0]+=(x[i]+k)*(x[i]+k)
        b[0] = 1.-np.exp(-sums[0])
        return b
    
    def gradient(self,x):
        return pg.estimate_gradient(lambda x: self.fitness(x), x)
        
    def get_nobj(self):
        return 2
    
    def get_bounds(self):
        return ([-4] * self.dim,[4]*self.dim)
      
    def get_name(self):
        return "Fonseca Flaming Test Function - C++"
    
    def get_extra_info(self):
        return "Dimensions " + str(self.dim)


def optimize_arch(popsize=10, generations=1, islands=1, 
                  evolutions=1,func=fonseca_fleming, dim=3):
    """
    Function to optimize the given problem. 
    Parameters can change to see their influence.
    """ 
    info="Prob= {}".format(func(dim).get_name())+"\n"+\
            "dim={} Pop={}, Gen={}, Isl={}, Evs={}".format(dim,popsize,generations,islands,evolutions)
    print(info)
    
    start=time.time()
    prob=pg.problem(func(dim))
    """Start optimization process."""
    # set up algorithm and optimize
    algo = pg.algorithm(pg.nspso(gen=generations))
    archi = pg.archipelago(islands, algo=algo, prob=prob, pop_size=popsize)
    # evolution
    set_time=time.time()
    archi.evolve(evolutions)
    archi.wait_check()
    print("Build time :{}ms".format(round(1000*(set_time-start),2)))
    print("Calculation time :{}ms".format(round(1000*(time.time()-set_time),2)))
    
    #Get Data on final population
    fits_log, vectors_log = [], []
    vectors = [isl.get_population().get_x() for isl in archi]
    vectors_log.append(vectors)
    fits = [isl.get_population().get_f() for isl in archi]
    fits_log.append(fits)
    table_fits=[[f[0], f[1]] for fit in fits for f in fit] 
    
    #Plot the whole population to see the last evolution
    df = pd.DataFrame(table_fits)
    ax = df.plot(kind='scatter', x=0, y=1, grid=True)
    ax.set_xlabel('f1(x)')
    ax.set_ylabel('f2(x)')
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)
    ax.set_title(info)
    #Save result to a file
    name= func(dim).get_name()+" "+func(dim).get_extra_info()
    #plt.savefig(name+".jpg")
    

if __name__ == '__main__':
    '''
    total population = popsize*islands
    optimize_arch(popsize, generations, islands, evolutions, func, dim)
    '''
    #optimize_arch(100,5,20,3,fonseca_fleming,3)
    #optimize_arch(100,8,15,3,fonseca_fleming,3)
    #optimize_arch(50,10,15,3,fonseca_fleming,3)
    optimize_arch(25,20,10,3,fonseca_fleming,3) #73+1797 ms
    optimize_arch(25,20,10,3,fonseca_fleming_c,3) #6+62 ms
    plt.show()
    #Both figures show the same Pareto Front. 
