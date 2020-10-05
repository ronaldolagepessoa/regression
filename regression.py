import pandas as pd
import plotly.express as px
import numpy as np




def regplot(x, y, model=None):
    fig = px.scatter(x=x, y=y)
    if model is not None:
        fig.add_trace(px.line(x=x, y=model.predict(x.values.reshape(-1, 1))).data[0])
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
    print(regplot(df.x, df.y))