import numpy as N
import scipy.io

def load_city_dist_matrix():
    return scipy.io.loadmat("dataset/city_dist_matrix.mat")["dist"]
    
def compute_transporation_matrix(city_dist, city_pop):
    t_mat = N.outer(1/N.maximum(city_pop,1), city_pop) / N.maximum(city_dist,1) 
    N.fill_diagonal(t_mat, 0)
    return t_mat

def create_transfer_matrix(city_dist, city_pop, mobility=0.001, mobility_pop=0.05, t_mat=None):
    """
    mobility: tendency to leave a city
    mobility_pop: tendency to leave a city due to population
    """
    #assert 0 <= mobility     and mobility <= 1
    assert 0 <= mobility_pop and mobility_pop <= 1
    
    n = city_pop.shape[0]
    assert city_dist.shape == (n, n)
    
    city_pop = N.maximum(city_pop, 1)
    stay = mobility * (1 - mobility_pop) * city_pop / city_pop.sum()
    assert N.all(stay >= 0)
    
    if t_mat is None:
        t_mat = compute_transporation_matrix(city_dist, city_pop)
        
    prob = (1 - stay)[:,N.newaxis] * t_mat / t_mat.sum(axis=1)[:,N.newaxis]
    prob[range(n),range(n)] = stay
    
    return prob