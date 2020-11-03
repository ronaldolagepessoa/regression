from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import plotly.graph_objects as go



class Regression:

    def __init__(self, DataFrame):
        self.df = DataFrame.dropna(axis=0)
        self.y = None
        self.X= None

    def set_y(self, y_column):
        self.y_column = y_column
        self.y = self.df[y_column]
    
    def set_X(self, drop_columns=None, include_columns=None):
        self.onehot = OneHotEncoder(sparse=False, drop="first")
        self.mmscaler = MinMaxScaler()
        if drop_columns is None and include_columns is None:
            self.X = self.df.drop(self.y_column, axis=1).copy()
        elif drop_columns is not None:
            self.X = self.df.drop([self.y_column] + drop_columns, axis=1).copy()
        else:
            self.X = self.df.drop([self.y_column], axis=1).copy()
            self.X = keep_cols(self.X, include_columns)
        
        X_bin = self.onehot.fit_transform(self.X.select_dtypes(include=['object']))
        X_num = self.mmscaler.fit_transform(self.X.select_dtypes(exclude=['object']))
        self.X_all = np.append(X_num, X_bin, axis=1)
        
    def fit(self, kind='linear', C=1, grau=1, alfa=1, n_vizinhos=5, profundidade=5, 
    n_arvores=100, camadas=(100,), iteracoes=200):
        func = model_dict[kind]
        kwargs_set = {
            'C': C,
            'grau': grau,
            'alfa': alfa,
            'n_vizinhos': n_vizinhos,
            'profundidade': profundidade,
            'n_arvores': n_arvores,
            'camadas': camadas
        }
        self.model = func(self.X_all, self.y, kwargs_set)

    def fit_test(self, kind='linear', C=1, grau=1, alfa=1, n_vizinhos=5, profundidade=5, 
    n_arvores=100, camadas=(100,), iteracoes=200, train_size=2/3, random_state=1):
        X_train, X_test, y_train, y_test = train_test_split(self.X_all, 
        self.y, train_size=train_size, random_state=random_state)
        func = model_dict[kind]
        kwargs_set = {
            'C': C,
            'grau': grau,
            'alfa': alfa,
            'n_vizinhos': n_vizinhos,
            'profundidade': profundidade,
            'n_arvores': n_arvores,
            'camadas': camadas
        }
        model = func(X_train, y_train, kwargs_set)
        if len(self.X_all.shape) == 1:
            return model.score(X_test.reshape(-1, 1), y_test)
        else:
            return model.score(X_test, y_test)
         
    def predict(self, values):
        X_predict = pd.DataFrame(values, columns=self.X.columns)
        X_bin = self.onehot.transform(X_predict.select_dtypes(include=['object']))
        X_num = self.mmscaler.transform(X_predict.select_dtypes(exclude=['object']))
        X_all = np.append(X_num, X_bin, axis=1)
        return self.model.predict(X_all)
              
    def score(self):
        if len(self.X_all.shape) == 1:
            return r2_score(self.y, self.model.predict(self.X_all.reshape(-1, 1)))
        else:
            return r2_score(self.y, self.model.predict(self.X_all))

    def plot(self, reg_line=False):
        if self.X_all.shape[1] > 1:
            raise 'Mais de duas dimensões! Impossível de plotar.'
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=self.X_all.reshape(1, -1)[0], y=self.y.values, mode='markers', name='dados'))
            if reg_line:
                fig.add_trace(go.Scatter(x=self.X_all.reshape(1, -1)[0], 
                y=self.model.predict(self.X_all), 
                    mode='lines', name='regressão'))
                fig.update_layout(title='Dados + Regressão')
                return fig
            else:
                fig.update_layout(title='Dados')
            return fig

def keep_cols(DataFrame, keep_these):
    """Keep only the columns [keep_these] in a DataFrame, delete
    all other columns. 
    """
    drop_these = list(set(list(DataFrame)) - set(keep_these))
    return DataFrame.drop(drop_these, axis = 1)

def reg_linear(X, y, kwargs_set):
    model = LinearRegression()
    if len(X.shape) == 1:
        model.fit(X.values.reshape(-1, 1), y)
    else:
        model.fit(X, y)
    return model


def reg_rbf(X, y, kwargs_set):
    model = SVR(kernel='rbf', C = kwargs_set['C'])
    if len(X.shape) == 1:
        model.fit(X.values.reshape(-1, 1), y)
    else:
        model.fit(X, y)
    return model


def reg_poly(X, y, kwargs_set):
    model = SVR(kernel='poly', degree = kwargs_set['grau'])
    if len(X.shape) == 1:
        model.fit(X.values.reshape(-1, 1), y)
    else:
        model.fit(X, y)
    return model


def reg_krr(X, y, kwargs_set):
    model = KernelRidge(alpha = kwargs_set['alfa'])
    if len(X.shape) == 1:
        model.fit(X.values.reshape(-1, 1), y)
    else:
        model.fit(X, y)
    return model


def reg_knn(X, y, kwargs_set):
    model = KNeighborsRegressor(n_neighbors=kwargs_set['n_vizinhos'])
    if len(X.shape) == 1:
        model.fit(X.values.reshape(-1, 1), y)
    else:
        model.fit(X, y)
    return model


def reg_gauss(X, y, kwargs_set):
    model = GaussianProcessRegressor()
    if len(X.shape) == 1:
        model.fit(X.values.reshape(-1, 1), y)
    else:
        model.fit(X, y)
    return model


def reg_arvore(X, y, kwargs_set):
    model = DecisionTreeRegressor(max_depth=kwargs_set['profundidade'])
    if len(X.shape) == 1:
        model.fit(X.values.reshape(-1, 1), y)
    else:
        model.fit(X, y)
    return model


def reg_floresta(X, y, kwargs_set):
    model = RandomForestRegressor(n_estimators=kwargs_set['n_arvores'], 
    max_depth=kwargs_set['profundidade'])
    if len(X.shape) == 1:
        model.fit(X.values.reshape(-1, 1), y)
    else:
        model.fit(X, y)
    return model


def reg_neural(X, y, kwargs_set):
    model = MLPRegressor(hidden_layer_sizes=kwargs_set['camadas'], 
    max_iter=kwargs_set['iteracoes'])
    if len(X.shape) == 1:
        model.fit(X.values.reshape(-1, 1), y)
    else:
        model.fit(X, y)
    return model


model_dict = {
    'linear': reg_linear, 
    'rbf': reg_rbf,
    'poly': reg_poly,
    'krr': reg_krr,
    'knn': reg_knn,
    'gauss': reg_gauss,
    'arvore': reg_arvore,
    'floresta': reg_floresta,
    'neural': reg_neural}

if __name__ == '__main__':
    import pandas as pd
    # df = pd.read_csv('airbnb_ny2.csv')
    x = np.linspace(0, 10, 100)
    df = pd.DataFrame({'x': x})
    y = 3 * x + 5 + np.random.normal(0, 2, 100)
    df['y1'] = y
    y = -3 * x + 5 + np.random.normal(0, 2, 100)
    df['y2'] = y
    y = 3 * x + 5 + np.random.normal(0, 5, 100)
    df['y3'] = y
    y = 2 ** x  - x ** 3 + np.random.normal(0, 8, 100)
    df['y4'] = y
    # df.to_csv('reg1.csv', index=False)
    model = Regression(df)
    # print(model.df.columns)
    model.set_y('y4')
    model.set_X(include_columns=['x'])

    # print(model.X.columns)
    model.fit(kind='neural', camadas=(100,))
    # print(model.predict([[100, 2, 2, 1, 10, 'aceita', 800, 1500, 150, 80]]))
    # print(model.score())
    # print(model.fit_test(kind='arvore', n_arvores=70, profundidade=7))
    fig = model.plot(reg_line=True)
    fig.show()


