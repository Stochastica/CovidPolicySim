import matplotlib.pyplot as pyplot
import seaborn as S
import pandas as P
import numpy as N

def create_df_from_result(r):
    d = P.DataFrame(N.array(r).reshape(-1, 6),
                    columns=['iterations', 'susc', 'latent',
                             'infect', 'asympt', 'recov'])
    d['pop'] = d['susc'] \
        + d['latent'] + d['infect'] \
        + d['asympt'] + d['recov']
    return d

def plot_history(r):
    df = create_df_from_result(r)
    S.lineplot(x='iterations', y='value', hue='variable',
           data=df.melt(id_vars='iterations'))