from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score



def predict(X, model):
    try: 
        if len(X.shape) == 1:
            return model.predict(X.values.reshape(-1, 1))[0]
        else:
            return model.predict(X.values)[0]
    except:
        return model.predict([X])[0]


def score(X, y, model):
    if len(X.shape) == 1:
        return r2_score(y, model.predict(X.values.reshape(-1, 1)))
    else:
        return r2_score(y, model.predict(X))


def reg_linear(X, y):
    model = LinearRegression()
    if len(X.shape) == 1:
        model.fit(X.values.reshape(-1, 1), y)
    else:
        model.fit(X, y)
    return model


def reg_svm(X, y, kernel='rbf'):
    """
    kernel = "linear" ou "rbf". valor padrão igual a "rbf"
    """
    model = SVR(kernel=kernel)
    if len(X.shape) == 1:
        model.fit(X.values.reshape(-1, 1), y)
    else:
        model.fit(X, y)
    return model


def reg_knn(X, y, n_vizinhos=5):
    model = KNeighborsRegressor(n_neighbors=n_vizinhos)
    if len(X.shape) == 1:
        model.fit(X.values.reshape(-1, 1), y)
    else:
        model.fit(X, y)
    return model


def reg_gauss(X, y):
    model = GaussianProcessRegressor()
    if len(X.shape) == 1:
        model.fit(X.values.reshape(-1, 1), y)
    else:
        model.fit(X, y)
    return model


def reg_arvore(X, y, profundidade=5):
    model = DecisionTreeRegressor(max_depth=profundidade)
    if len(X.shape) == 1:
        model.fit(X.values.reshape(-1, 1), y)
    else:
        model.fit(X, y)
    return model


def reg_floresta(X, y, n_arvores=100, profundidade=6):
    model = RandomForestRegressor(n_estimators=n_arvores, max_depth=profundidade)
    if len(X.shape) == 1:
        model.fit(X.values.reshape(-1, 1), y)
    else:
        model.fit(X, y)
    return model


