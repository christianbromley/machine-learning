# Reinforcement learning

How to match a state (s) to an action (a)?

- where supervised learning does not work
- need a reward function
  - does something good +1
  - does something bad -100
- is a Markov Decision Process (future only depends on current state)

Core elements:
- s: state
- a: action
- R(s): reward
- s': new state
- gamma: discount factor (normally 0.9 - or other close to 1)

Return:
- the sum of different sets of rewards weighted by a set of discount factors
- R1 + gamma*R2 + (gamma**2)*R3 + (gamma**3)*R3

Policy pi:
- takes as input (s)
- map to action (a) via policy pi
- if in state 2: perform action 
- if in state 3: perform different action
- goal of RL is to find a poliy that tells you what action to take in every state so as to maximise return

State action value or Q function:
- Q(s,a)
  - Policy example
    - start in state s
    - take action a
    - then behave optimally afterwards