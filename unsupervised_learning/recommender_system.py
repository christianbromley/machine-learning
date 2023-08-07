import tensorflow as tf

# Collaborative Learning

## items, users, ratings
## note very good at cold start problem - a new item with no ratings

## performing gradient descent - auto diff

## implementation without the sequential dense layers and model.compile model.fit

def collaborative_learning_algo(algorithm: str = 'gradient'):

    if algorithm == 'gradient':

        w = tf.Variable(3)  # we tell tf that w is a parameter
        x = 1
        y = 1  # target value
        alpha = 0.01  # learning rate

        iterations = 20

        for iter in range(iterations):
            # gradient tape to record steps
            # telling it how to compute the cost function J
            with tf.GradientTape() as tape:
                fwb = w*x
                j = (fwb - y)**2

            # calculate the gradients of the cost J to enable diff
            [djdw] = tape.gradient(j, [w])

            # update parameter w
            w.assign_add(-alpha * djdw)

    elif algorithm == 'adam':
        # instantiate
        optimiser = kras.optimizers.Adam(learning_rate=1e-1)

        # define number of iterations
        iterations = 100

        for iter in range(iterations):
            with tf.GradientTape() as tape:
                # compute the cost
                cost_val = cofiCostFuncV(X, W, b, Ynorm, R, num_users, num_movies, lambda_val)

                # automatically retrieve the gradients of the trainable variables with response to loss
                grads = tape.gradient(cost_val, [X,W,b])

                # run a step of gradient descent by updating the value of the variables to minimise the loss
                optimiser.apply_gradient(zip(grads, [X,W,b]))

    return




# Content based filtering algorithm

## Tensorflow implementation

def content_based_filtering():

    user_nn = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=32)
    ])

    item_nn = tf.keras.models.Sequential([
        tf.keras.layers.Dense(units=256, activation='relu'),
        tf.keras.layers.Dense(units=128, activation='relu'),
        tf.keras.layers.Dense(units=32)
    ])

    # create user input and feed it the user network
    input_user = tf.keras.layers.Input(shape=(n_user_features))
    vu = user_nn(input_user)
    vu = tf.linalg.l2_normalize(vu, axis=1)

    # create item input and feed it the item network
    input_item = tf.keras.layers.Input(shape=(n_item_features))
    vm = item_nn(input_item)
    vm = tf.linalg.l2_normalize(vm, axis=1)

    # determine similarity of two vectors
    output = tf.keras.layers.Dot(axes=1)([vu,vm])

    # specify inputs and outputs of the model
    model = Model([input_user, input_item], output)

    # specify cost function
    cost_fn = tf.keras.losses.MeanSquaredError()

    return model, cost_fn