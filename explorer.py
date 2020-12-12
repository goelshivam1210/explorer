'''
The goal of this class is to find the state space 
initialize the new environment
modify the network
initialize the weights with similarity to the exoistoing object weights

use sampled trajectories to generate optimal action set
use the optimal action set to perform reasoning over exploration over current 
observed states.

also add rewards based on developing agent to be more curious to even
visit novel states

run this to evaluate 
$ python explorer.py # need to add stuff

For questions contact
shivam.goel@tufts.edu
# Michael add email here

'''

class Explorer(object):
    def __init__(parameter_list):
        pass


    def get_optimal_actions(self, ):
        pass
    '''
    I/P: sampled trjectory
    O/P: ranked actions on novel states type the agent should preferably take
    '''

    def adapt_network(self, ):
        """
        docstring
        """
        pass
    '''
    I/P: new environment specifics, old environment specifics, learned_model
    O/P: modified network flag= True/False
    '''

    def adapt_network(self, parameter_list):
        """
        I/P: new environment specifics, old environment specifics, learned_model
        O/P: modified network flag= True/False
        """
        pass

