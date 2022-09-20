import matplotlib.pyplot as plt
import numpy as np
import quantstats as qs


def portfolio_backtest(weight_vector, securities_vector, daily_rebalance=False, title=None, tearsheet=False, mode='basic'):
    weight_vector_n = weight_vector.reindex(weight_vector.index.union(securities_vector.index))
    weight_vector_n = weight_vector_n[weight_vector.index[0]:]
    return_frame = np.exp(np.log(securities_vector).diff()) - 1
    return_frame = return_frame[weight_vector.index[0]:]
    if daily_rebalance:
        weight_vector_n = weight_vector_n.ffill()
    else:
        for p in range(len(weight_vector_n)):
            if np.isnan(weight_vector_n.iloc[p, 1]):
                new_weights_unnormalized = weight_vector_n.iloc[p - 1, :] * (return_frame.iloc[p - 1, :] + 1)
                weight_vector_n.iloc[p, :] = new_weights_unnormalized / sum(new_weights_unnormalized)

    weighted_returns = weight_vector_n.shift(1) * return_frame
    port_returns = np.sum(weighted_returns,
                          axis=1)  # shift so that we are using the weights we had over that period rather than the ones we found with hindsight
    port_returns = port_returns[weight_vector.index[0]:]
    total_return = (port_returns + 1).cumprod()
    total_return.plot()
    if tearsheet:
        qs.reports.metrics(port_returns, mode=mode)
    plt.title('Cumulative Return')
    if title is not None:
        plt.title(title)
        plt.savefig(title)
    return total_return, np.std(port_returns)
