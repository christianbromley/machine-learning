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

Bellman equation:
Q(s,a) (return under a set of functions)
s' = state you get to
a' = action you use to get to s'
Q(s,a) = R(s) + gamma* max a' * Q(s',a')

Continuous states:
- s may contain several features
- e.g. if it is a car:
  - x = x location
  - y = y coord
  - theta = angle
-  these numbers are inputs to a policy which then decides the next move

Can train a neural network to approximate the state value action function
- predicts Q(s,a)
-  encode as set of 12 numbers into a neural network
- predicting a state action pair for each of the possible actions
- typically you build a big training set with different values of x and y
- neural net takes Q(s,a) or x and predicts Rs+gamma*maxaQ(s',a') - bellman equation
- take random actions in a simulator in order to observe what happens with different actions
  - s1,a1,Rs1,s'1 <- compute training tuples
    - s1 = 8 numbers or features state of lunar 
    - a1 = 4 numbers or one hot encoding of what the action was
    - compute y1 from these values using bellman

Full learning algorithm
- initialise neural network randomly as guess of Q(s,a)
  - random weights
- in a loop take random actions to get tuples of (s,a,Rs,s') and store 10,000 most recent
- create training set by computing x and y
  - x = (s,a)
  - y = Rs + gamma*maxa Q(s',a')
- Qnew(s,a) learns to approximate y

epsillon-greedy policy:
- in some state s:
- Option 1
  - pick an action that maximizes Q(s,a)
- Option 2
  - with prob 0.95, pick the action a that maximises Q(s,a) (exploitation)
  - 5% of time pick an action a randomly (exploration)
- here epsillon greedy policy = 0.05
- sometimes you start with a high epsillon and decrease it over time therefore over time you are using more the knowledge you have learned


Mini-batches:
- imagine you have a very large training set e.g. m=100million
- every step of gradient descent requires computing the average over 100million samples to compute J(w,b) loss function
- mini batch gradient descent uses a subset of the 100million e.g. m' = 1000
- the second iteration of the algo would look at the second mini batch unique from mini batch 1
- in contrast to batch learning mini-batch learning may not always go in the right direction of gradient descent so the line towards the global minimum is more wiggly compared to batch learning
- much faster algorithm when you have a very large training set

Soft-updates:
- Q (w,b) = Qnew (Wnew, Bnew) -> we don't want to compltely override the old neural network as it might get worse
- with a soft update you set w = 0.01Wnew + 0.99W
- with a soft update you set b = 0.01Bnew + 0.99B
- allows you to make more gradual changes to the neural network parameters
- reinforcement learning algo converges more reliably with soft updates

Experience-replay:
- if agent tries to learn from consecutive experiences it can run into problems due to strong correlations between them
- experience replay allows to generate uncorrelated experiences for training the agent
- store the states, actions and rewards in a memory buffer and then sample a random minibatch
- 