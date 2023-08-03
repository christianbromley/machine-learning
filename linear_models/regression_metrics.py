def mse(actual, predicted):
    m = len(actual)
    err = 0
    for i in range(m):
        err += ((predicted[i] - actual[i]) ** 2)
    return err / (2 * m)