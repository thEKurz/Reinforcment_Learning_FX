import pypfopt as pyp
import pandas as pd
import cvxpy as cp

def markowitz_rebalance(securities_vector,recalc_period, prior_period, obj='minvar', verbose=True,
                        weight_bounds=(0, 1), static_start_date=None,returns_data=False, cov_mat = None, mu=None):
    weights_list = []
    dates = []
    if static_start_date is not None:
        l = len(securities_vector[static_start_date:])
        securities_vector = securities_vector[-(prior_period+l):]
    if recalc_period is None:
        prior_vector = securities_vector[:prior_period]
        if mu is not None:
            mu = mu
        else:
            mu = pyp.expected_returns.mean_historical_return(prior_vector, returns_data=returns_data)
        if cov_mat is not None:
            S = cov_mat
        else:
            S = pyp.risk_models.CovarianceShrinkage(prior_vector, returns_data=returns_data).ledoit_wolf()
        ef = pyp.efficient_frontier.EfficientFrontier(mu, S, weight_bounds=weight_bounds)
        if obj == 'minvar':
            weights = ef.min_volatility()
        if obj == 'sharpe':
            weights = ef.max_sharpe()
        cleaned_weights = ef.clean_weights()
        weights_list.append(cleaned_weights)
        dates.append(prior_vector.index[-1])
    else:
        i = 1
        i_ = round((len(securities_vector) - prior_period) / recalc_period)
        for p in range(0, len(securities_vector) - prior_period, recalc_period):

            prior_vector = securities_vector[p:prior_period + p]

            if verbose:
                print("Iteration " + str(i) + " of " + str(i_))
                print(str(prior_vector.index[0]) + " - " + str(prior_vector.index[-1]))
            if mu is not None:
                mu=mu
            else:
                mu = pyp.expected_returns.mean_historical_return(prior_vector,returns_data=returns_data)
            if cov_mat:
                S = cov_mat
            else:
                S = pyp.risk_models.CovarianceShrinkage(prior_vector,returns_data=returns_data).ledoit_wolf()
            ef = pyp.efficient_frontier.EfficientFrontier(mu, S, weight_bounds=weight_bounds)
            if obj == 'minvar':
                weights = ef.min_volatility()
            if obj == 'sharpe':
                weights = ef.max_sharpe()
            cleaned_weights = ef.clean_weights()
            weights_list.append(cleaned_weights)
            dates.append(prior_vector.index[-1])
            i += 1
    weight_df = pd.DataFrame(weights_list)
    weight_df.index = dates

    return weight_df

def factor_port(sigma, factor_betas, betas: list = [1]):
    w = cp.Variable(len(sigma.columns))
    obj = cp.quad_form(w,sigma)
    constraints = [cp.sum(w)==1,w >= 0]
    if len(betas)>1:
        for p in range(len(betas)):
            constraints += [w @ factor_betas.iloc[:,p] == betas[p]]
    else:
        constraints += [w @ factor_betas == betas[0]]
    prob= cp.Problem(cp.Minimize(obj), constraints)
    prob.solve()
    weights = pd.DataFrame(w.value, index=sigma.index)
    return weights