import pandas as pd
import plotly.graph_objects as go
import numpy as np



def regplot(x, y, model=None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers', name='dados'))
    if model is not None:
        fig.add_trace(go.Scatter(x=x, y=model.predict(x.values.reshape(-1, 1)), 
        mode='lines', name='regressão'))
        fig.update_layout(title='Dados + Regressão')
    else:
        fig.update_layout(title='Dados')
    return fig


if __name__ == "__main__":
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


    from models import reg_floresta, score, predict

    model = reg_floresta(df.x, df.y4)
    print(score(df.x, df.y4, model))
    print(predict(df.x, model))
    # regplot(x=df.x, y=df.y4, model=model).show()
        