import numpy as np

def bellman_equation(Rs, gamma, maxa):
    # Q(s,a) = R(s) + gamma* max a' Q(s', a')
    # R(s) in this is the reward you get immediately
    # gamma* max a' Q(s', a') is the reward you get from behaving optimally
    # R1 + gamma*R2 + gamma**2*R3
    return Rs + (gamma*maxa)


[0,0,0,0,100]
def compute_return(reward_sequence: list, gamma: float):

    calculated_return = 0
    for i, val in enumerate(reward_sequence):
        if i == 0:
            calculated_return += val
        else:
            calculated_return += ((gamma**i)*val)
    print(calculated_return)
    return calculated_return

#X = np.array([[0,0,0,0,100], [0,0,0,0,50]])
def expected_return(reward_sequences: np.array, gamma: float):
    returns = []
    for i in range(len(reward_sequences)):
        print(reward_sequences[i])
        sequence_return = compute_return(reward_sequence=reward_sequences[i], gamma=gamma)
        returns += [sequence_return]

    expected_return = np.mean(returns)
    return expected_return

#print(expected_return(X, gamma=0.9))