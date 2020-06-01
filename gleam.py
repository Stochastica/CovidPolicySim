import numpy as N
import numpy.linalg as NL
import numpy.random as NR
import pandas
import progressbar
import pathos.multiprocessing as Mp

class Gleam:
    
    
    def __init__(self, pop, transfer_func, pressure_func, **kwargs):
        """
        pop: A (states, cells) matrix
        transfer_func: A [date -> (state, cells, cells)] function,
            the returned matrix must be a Markov matrix on the last two dims
        pressure_func: A [(date, cell) -> (state,state)] function,
            producing a markov matrix
        """
        self.pop = pop
        self.transfer_func = transfer_func
        self.pressure_func = pressure_func
        
        self.n_states = self.pop.shape[0]
        self.n_cells = self.pop.shape[1]
        
        self.rate_birth = kwargs.get('rate_birth', 0.011)
        self.rate_death = kwargs.get('rate_death', self.rate_birth)
        
        self.n_threads = kwargs.get('n_threads', 1)
        if self.n_threads != 1:
            self.pool = Mp.ProcessingPool(self.n_threads)
        
        
        
    def timestep(self, t, state_diag_func=None, policy_func=None):
        assert N.all(self.pop >= 0)
        pop_shape = self.pop.shape
        
        policy = policy_func(t, self.pop) if policy_func else None
        
        # Calculate population transfer
        
        # p[t,q,i] -> p[t,q,1..k] = multinomial(p[t,q,i], transfer[q,i,1..k])
        #
        
        transfer_mat = self.transfer_func(t, policy)
        
        if self.n_threads != 1:
            def transmission_calc(i):
                pages = [NR.multinomial(self.pop[q, i], transfer_mat[q, i, :])[N.newaxis,:]
                         for q in range(self.n_states)]
                pages = N.concatenate(pages, axis=0)
                assert pages.shape == (self.n_states, self.n_cells)
                return pages[..., N.newaxis]
            flow = self.pool.map(transmission_calc, range(self.n_cells))
            flow = N.concatenate(flow, axis=-1)
            assert flow.shape == (self.n_states, self.n_cells, self.n_cells)
            self.pop = N.sum(flow, axis=-1)
        else:
            flow = N.zeros((self.n_states, self.n_cells, self.n_cells), dtype='int')
            for q in range(self.n_states):
                for i in range(self.n_cells):
                    flow[q,i,:] = NR.multinomial(self.pop[q, i], 
                                                 transfer_mat[q, i, :])

            self.pop = N.sum(flow, axis=1)
            
        # Calculate birth and death
        self.pop[0,:] += NR.binomial(N.sum(self.pop, axis=0), self.rate_birth / 365)
        self.pop[:,:] -= NR.binomial(self.pop,                self.rate_death / 365)
        
        # Evolve state
        
        if self.n_threads != 1:
            def mobility_calc(i):
                pressure = self.pressure_func(t, i, self.pop[:,i], policy)

                flow = N.zeros((self.n_states, self.n_states))
                for q in range(self.n_states):
                    try:
                        flow[q,:] = NR.multinomial(self.pop[q, i], pressure[q, :])
                    except:
                        print(pressure)
                        raise
                return N.sum(flow, axis=0)[:,N.newaxis]
            result = self.pool.map(mobility_calc, range(self.n_cells))
            self.pop = N.concatenate(result, axis=-1)
        else:
            for i in range(self.n_cells):
                pressure = self.pressure_func(t, i, self.pop[:,i], policy)

                flow = N.zeros((self.n_states, self.n_states))
                for q in range(self.n_states):
                    try:
                        flow[q,:] = NR.multinomial(self.pop[q, i], pressure[q, :])
                    except:
                        print(pressure)
                        raise
                self.pop[:,i] = N.sum(flow, axis=0)
        
        s = list(N.sum(self.pop, axis=1))
        if state_diag_func is None:
            return s, None
        else:
            return s, state_diag_func(t, self.pop, policy)
        
        
        
    def simulate(self, n_timesteps,
                 initial_timestep=0, verbose=False,
                 state_diag_func=None, policy_func=None):
        hist = [[initial_timestep] + list(N.sum(self.pop, axis=1))]
        diags = []
        
        time_range = range(initial_timestep,n_timesteps)
        if verbose:
            time_range = progressbar.progressbar(time_range)
        for t in time_range:
            s, diag = self.timestep(t, state_diag_func, policy_func)
            hist.append([t+1] + s)
            diags.append(diag)
        return hist, diags