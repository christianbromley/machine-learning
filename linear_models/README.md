# Linear modelling

## Notation
First some notation:

f = function (model)
i = row
x = input
y = output
x(i) = input row
yhat = prediction for y
f(x) = function to comute from x
m = number of samples

cost function - MSE

J = 1/2m sum ( (yhat(i) - y(i))**2 )

minimise J (w, b)


## Gradient descent
- pick direction of steepest descent
- spin around, in which direction would I go to take the steepest step
- local minima: more complex algorithms can have multiple troughs in the cost function
- linear regression is always convex

Algorithm:
- new w = (w - alpha ) d/dw (Jw,b) (exact same for b as for w - the different parameters, must peform simultaneous update of the parameters)
- alpha is the learning rate (a small positive number e.g. 0.01 - how big a step you take down hill)
- d/dw derivative term of the cost function J -> in which direction you want to take the step

strt with value of w

w = w-alpha d/dw J(w)

draw a tangent
w = w-alpha*(positive number)
you decrease w (you have learnt)
derivative = change in y / change in x
as you get nearer the local minimum the steps get smaller because of the derivative even if you have a fixed learning rate
Batch gradient descent - each step lookng at all the training data


## Multiple regression
Xy = jth feature
n = number of features
x^i = becomes a list of numbers of a vector

f(x) = w1x1 + w2x2 + …+wnxn + b

warrow = [w1, w2, w3…wn] = vector of parameters/ chefs/ weights

simpler equation W-arrow . X-arrow + b (. = dot product)

Vectorisation = important for speeding up code, particularly with many features

Alternative to gradient descent = normal equation (only for linear regression) -> slow, solves w and b without iterations

Array[[row1], [row2], [row3]]

Make sure gradient descent is working:
- plot cost function against number of iterations (learning curve)
- when curve no longer decreases, this means that it has converged
- automatic convergence test: if epsilon = 0.001, if cost decreases by less than this in one iteration that you can declare convergence

Choose a good learning rate:
- with a small enough learning rate the cost function should decrease on every iteration
    e.g. just set learning rate to very small number to check


## Polynomial regression
features raised to power of 2 or 3 or any other power
function = w1X + w2x^2 + w3x^3 + b -> in this x is the same feature
you can also take the square root


## Logistic regression
log odds (t) = B1*X1 + B0

logistic function converts log odds into probability estimate

probability = 1/(1 + e^-t)

Wald test significantly associated with the probability of being a BEN2293 v placebo

Z-stat ratio of coef / standard error


## Classification
sigmoid or logistic function

outputs values between 0 and 1
1 / 1 + e^-z
when Z is high -> g(z) almost 1, when Z is low g(z) almost 0

straight line function
z = wx + b
g(z) = 1 / 1 + e^-z

put these together to get logistic regression model
f(x) = g(wx+b) = 1/1+e^-(wx+b)
outputs probability that the class label will be 1

where to set the decision boundary? commonly 0.5
f(x)>0.5 -> that = 1
when wx+b > 0 -> model predicts 1

decision boundary wx+b=0 -> x1 + x2 -3 (no coefficients)
x1+x3 = 3

non-linear decision boundaries
-> can use polynomials in logisitc regression

w1x1^2 + w2x2^2 + b

set coefs to 1

x1^2 + x2^2 = 1

cost function for logistic regression
need to make J(w,b) convex to avoid getting stucj in local minima
measures how well you’re doing on one training sample

if y=1 -log(fx)
if y=0 -log(1-fx)

if algorithm predicts a 1 at 0.5, -log(0.5) is the loss

When y=0 actually, then loss is -log(1-0)

simplified loss function:
there’s another way of representing the loss function:
simplified loss function = -ylog(fx) - (1-y)log(1-fx)

Cost is average loss
This is maximum likelihood estimation
Cost = -1/m sum of [ylog(fx) + (1-y)log(1-fx))]


## Gradient descent for logistic regression
- minimise the cost function
- cost function represented by J(w,b)
- w - alpha * derivative term
- simultaneous updates required
- only difference with linear regression is the actual f(x), but the rest of the gradient descent algo remains the same


## Overfitting and regularisation

high bias = underfitting (preconception of data being linear)
low bias = overfitting
Generalisation - fits new data well


Regularisation can help to address overfitting:
- collect more training data will help
- if not, then include/exclude features
- use only most relevant features / feature selection
- Disadvantage: throwing away features
- regularization reduces the impact of some of teh features without eliminating outright
- no need to regularise parameter b


Cost function for regularization:
add 1000xW3^2 to the loss function you are basically penalising those terms

linear regression most function

J(w,b) = 1/2m sum(fx - y)^2) + lambda/2m sum(wj^2)

some also include + lambda/2m * b^2

algorithm tries to keep the regularisation term small - lambda dictates balabce of data fit

lambda = 0 -> no regularisation -> overfit
Lambda = very large -> learning algorithm would pick coefficients very close to 0 =? Underfit
Choosing lambda is therefore important

gradient descent with regularisation

Wj = wj(1 - alpha(lambda/m) - (usual update of linear regression)

The first bit of the term tends to end up close to 1 -> so on every iteration your multiplying wj by a number just slightly less than 1 so you are shrinking the estimate

Usual: -alpha (1/m) sum (fx - y) Xj

Derivative calculation:
Dj_dw = 1/m sum( wx + b - y) Xj) +( lambda/m)wj


Regularised logistic regression:

- add the previous cost function for glm to (+ lambda/2m sum(wj^2) )
- simultaneous updates - derivative term gets addition term lambda/m x Wj
- no update to b
