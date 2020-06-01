import numpy as N
import numpy.linalg as NL
import numpy.random as NR
import multiprocessing

import gleam
import datagen
import losses

I_SUSC = 0
I_LATENT = 1
I_SYMPT = 2
I_ASYMPT = 3
I_RECOV = 4

HIST_BEGIN = 22
HIST_QUARANTINE_START = 84 - HIST_BEGIN # Emergency on Mar. 22, relative to 01/22
HIST_TRAIN_BEGIN = 49 # 100th case
HIST_TRAIN_LIMIT = 115
HIST_POLICY_BEGIN = 152 - HIST_BEGIN


# Following parameters are labeled (log) if their scale is logarithmic.
#

DEFAULT_PARAMETER = {
    # (log) The daily spread pressure of the virus
    # Range: < 0
    'infectiosity': 0,
    # The daily spread pressure reduction after quarantine is on
    # Range: [0,1]
    'infectiosity_reduct': -.3,
    # (log) The tendency for people to move.
    # Range: < 0 (should be a very small number like -2)
    'mobility': -1.0,
    # Range: [0,1]
    'mobility_reduct': -.3,
    # (log) Tendency for people to move due to population.
    # Range: < 0
    'mobility_pop': -1.3,
    # (log) The number of estimated latent cases based on the number of active cases
    # Range; > 0
    'latent_ratio': 1,
}

DEFAULT_POLICY = {
    # A city should open when
    #
    # infected_pop <= city_pop * thresh * (1 - city_pop/total_pop)^thresh_p * (1+thresh_t)^days
    
    # (log), thresh
    'thresh': -7.0,
    'thresh_p': 0.1,
    # (log) thresh_t
    'thresh_t': 0.0,
}

def to_categorical(x, n_categories):
    return N.eye(n_categories)[x]


class Simulacra:
    
    def __init__(self,
                 city_dist_matrix, city_province_map, city_pop, 
                 infect_history, infect_init,
                 loss_func=losses.log_mean_square_loss,
                 quarantine_loss=0.001,
                 n_states=5,
                 n_simulation_repeat=8,
                 n_threads=1):
        """
        city_dist_matrix: unit:distance
        city_province_map: 1d array mapping cities to provinces
        city_pop: unit:persons
        
        quarantine_loss: unit:1/day. The higher this value is the more the loss function favours re-opening over quarantine.
        """
        
        self.city_dist_matrix = city_dist_matrix
        self.city_province_map = city_province_map
        self.city_pop = city_pop
        self.n_provinces = city_province_map.max() + 1
        self.transportation_matrix = datagen.compute_transporation_matrix(self.city_dist_matrix, self.city_pop)
        
        self.loss_func = loss_func
        self.quarantine_loss = quarantine_loss
        self.n_states = n_states
        self.n_cells = self.city_pop.shape[0]
        
        # A (n_cities, n_provinces) map
        self.city_province_matrix = to_categorical(self.city_province_map, self.n_provinces)
        
        self.infect_history = infect_history
        self.infect_init = infect_init
        self.hist_test_limit = self.infect_history.shape[0]
        self.hist_policy_limit = 720
        
        self.n_simulation_repeat = 8
        self.n_threads = n_threads
    
    def evaluate(self, pop, initial_time, n_timesteps, **kwargs):
        
        verbose = kwargs.get('verbose', False)
        
        infectiosity   = N.power(10, kwargs['infectiosity'])
        infectiosity_q = infectiosity * N.power(10, kwargs['infectiosity_reduct'])
        #infectiosity_q = infectiosity * kwargs['infectiosity_reduct']
                                     
        mobility   = N.power(10, kwargs['mobility'])
        mobility_q = mobility * N.power(10, kwargs['mobility_reduct'])
        mobility_pop = N.power(10, kwargs['mobility_pop'])
        
        
        pressure_func = self.create_pressure_func(infectiosity, infectiosity_q)
        transfer_func = self.create_transfer_func(mobility, mobility_q, mobility_pop)

        diag_func = kwargs.get('diag_func', self.gleam_diagnostic)
        
        #print("Initial pop: {}".format(N.sum(pop, axis=1)))
        
        self.model = gleam.Gleam(pop, transfer_func, pressure_func, n_threads=self.n_threads)
        self.reset_policy_vector()
        hist,diags = self.model.simulate(n_timesteps,
                                    state_diag_func=diag_func, 
                                    policy_func=self.gleam_policy,
                                    initial_timestep=initial_time,
                                    verbose=verbose)
        
        y_pred, in_quarantine = zip(*diags)
        #print("y_pred: {}".format(len(y_pred)))
        #print("y_pred[0]: {}".format(y_pred[0].shape))
        return hist, N.concatenate([y[N.newaxis,:] for y in y_pred],axis=0).T, N.array(in_quarantine)
    
        
    def target_function_hist(self, **params):
        
        #kwargs = {**params, **DEFAULT_POLICY}
        self.set_policy(DEFAULT_POLICY)
        
        latent_ratio = N.power(10, params['latent_ratio'])
        pop = self.create_initial_pop(latent_ratio)
        y_true = self.infect_history[:,HIST_TRAIN_BEGIN:HIST_TRAIN_LIMIT]
        
        hist, y_pred, _ = self.evaluate(pop, HIST_TRAIN_BEGIN, HIST_TRAIN_LIMIT, **params)
        #y_pred = y_pred[:,1:]
        #print("y_true: {}, y_pred: {}".format(y_true.shape, y_pred.shape))
        
        assert y_true.shape == y_pred.shape
        return -self.loss_func(y_true, y_pred)
    
    
    def target_function_policy(self, **policy):
        if self.preset_params is None:
            raise RuntimError("Parameters are not initialised")
        
        verbose = policy.get('verbose', False)
        
        #kwargs = {**self.preset_params, **policy}
        self.set_policy(policy)
        hist, y_pred, in_quarantine = self.evaluate(self.policy_initial_pop,
                                                    HIST_POLICY_BEGIN, self.hist_policy_limit,
                                                    **self.preset_params, verbose=verbose)
        
        if verbose:
            print("y_pred={}, in_quarantine={}".format(y_pred.shape, in_quarantine.shape))
            
        hist = N.array(hist).T[1:,:]
        if False:
            # Compute healthcare system pressure
            infected_hist = hist[[I_SYMPT,I_ASYMPT],1:].sum(axis=0)

            assert in_quarantine.shape == (infected_hist.shape[0],)

            loss1 = infected_hist.max()
        else:
            # Compute total cases
            loss1 = hist[[I_LATENT, I_SYMPT, I_ASYMPT, I_RECOV],-1].sum()
            
        loss2 = in_quarantine.sum()
        if verbose:
            print("loss1={}, loss2={}".format(loss1, loss2))
            print("scale={}".format(loss2 / loss1))
        
        total_pop = hist[:,-1].sum()
        return -(loss1/total_pop + self.quarantine_loss*loss2/total_pop)
        #return -N.log10(loss1) - N.log10(loss2)
        #return -N.log10(loss1 + self.quarantine_loss * loss2)
        
    
    
    def set_params(self, p, pop):
        self.preset_params = p
        self.policy_initial_pop = pop
        
    def set_policy(self, policy):
        self.quarantine_thresh = N.power(10, policy['thresh'])
        self.quarantine_thresh_power = policy['thresh_p']
        self.quarantine_thresh_t = N.power(10, policy['thresh_t'])
    
    def gleam_diagnostic(self, t, pop, policy):
        """
        Convert population to provincial infection rate
        """
        pop_active = N.sum(pop[[I_SYMPT,I_ASYMPT,I_RECOV],:], axis=0)
        pop_active = pop_active@self.city_province_matrix
        
        assert pop_active.shape == (self.n_provinces,)
        
        # calculate how many people are in quarantine
        pop_quarantined = pop.sum(axis=0).dot(policy)
        
        return pop_active, pop_quarantined
    
    def gleam_identity_diagnostic(self, t, pop, policy):
        pop_quarantined = pop.sum(axis=0).dot(policy)
        return pop, pop_quarantined
    
    def reset_policy_vector(self):
        self.policy_enabled = N.ones((self.city_pop.shape[0],), dtype='bool')
        
    def gleam_policy(self, t, pop):
        """
        pop: entire population
        
        Return a boolean vector. true => quarantine, false => no quarantine
        """
        if t < HIST_QUARANTINE_START:
            return N.zeros((pop.shape[1],), dtype='bool')
        
        if t <= HIST_POLICY_BEGIN:
            return N.ones((pop.shape[1],), dtype='bool')
        
        pop_bycity = pop.sum(axis=0)
        pop_active = pop[[I_SYMPT,I_ASYMPT],:].sum(axis=0)
        pop_total = pop_bycity.sum()
        
        assert pop_bycity.shape == (pop.shape[1],)
        assert pop_active.shape == (pop.shape[1],)
        
        limits = pop_bycity * self.quarantine_thresh \
            * N.power(1 - pop_bycity/pop_total, self.quarantine_thresh_power) \
            * N.power(1 + self.quarantine_thresh_t, t)
        assert pop_bycity.shape == limits.shape
        self.policy_enabled  = (pop_active > limits) & self.policy_enabled
        return self.policy_enabled
        
                         
    
    def create_initial_pop(self, latent_ratio, seed=0):
        """
        Create population on day 49
        """
        NR.seed(seed)
        pop = N.zeros((self.n_states,self.n_cells), dtype='int')
        pop[I_SUSC,:] = self.city_pop
        pop[I_SYMPT,:] = self.infect_init // 5
        pop[I_ASYMPT,:] = self.infect_init - (self.infect_init // 5)
        pop[I_LATENT,:] = self.infect_init * latent_ratio
        
        return pop
    
    def create_pressure_func(self, infectiosity, infectiosity_q):
        
        assert 0 <= infectiosity and infectiosity <= 1
        assert 0 <= infectiosity_q and infectiosity_q <= 1
        
        def f(t, i, s, policy):

            incubation = 1/10
            recovery_mild = 1/20 #1/14
            recovery_severe = 1/30

            total = N.sum(s)
            if N.sum(s) == 0:
                beta = 0
            elif policy[i]:
                # quarantined
                beta = infectiosity_q * (s[I_LATENT] + s[I_ASYMPT]) / total
            else:
                beta = infectiosity * (s[I_LATENT] + s[I_SYMPT] + s[I_ASYMPT]) / total

            p = N.zeros((self.n_states, self.n_states))
            p[I_SUSC,I_SUSC] = 1 - beta
            p[I_SUSC,I_LATENT] = beta
            p[I_LATENT,I_LATENT] = 1 - incubation
            
            p[I_LATENT,I_SYMPT] = incubation * 0.2
            p[I_LATENT,I_ASYMPT] = incubation * 0.8

            p[I_SYMPT, I_SYMPT] = 1 - recovery_severe
            p[I_SYMPT, I_RECOV] = recovery_severe
            p[I_ASYMPT, I_ASYMPT] = 1 - recovery_mild
            p[I_ASYMPT, I_RECOV] = recovery_mild
            p[I_RECOV, I_RECOV] = 1

            return p
        return f
    
    def create_transfer_func(self, mobility, mobility_q, mobility_pop):
        
        def f(t, policy):
            vec_mobility = N.where(policy, mobility_q, mobility)
            mat = datagen.create_transfer_matrix(self.city_dist_matrix, self.city_pop,
                                                 mobility=vec_mobility, mobility_pop=mobility_pop)
            return N.repeat(mat[N.newaxis,...], self.n_states, axis=0)
        
        return f
        