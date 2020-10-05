from sklearn.linear_model import LinearRegression


def reg_linear(X, y):
    model = LinearRegression()
    if len(X.shape) == 1:
        model.fit(X.values.reshape(-1, 1), y)
    else:
        model.fit(X, y)
    return model