'''
Statustics and Data Analysis
Solutions to L14_Missing_data.pdf
'''

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
from scipy.optimize import minimize

N = 100
x = np.arange(1, N+1)
xm = 44
mask = np.arange(xm-2, xm+15)

def f_data_generating(x, xm, mask, N):

    k = 6000
    
    shape = 50
    expected_val = k * gamma.pdf(x, a=shape, scale=xm/shape)
    
    mu = 10
    sigma2 = 10
    z = np.random.normal(mu, np.sqrt(sigma2), N)
    
    val = np.round(expected_val + z).astype(float)
    val[mask] = np.nan
    
    val[val > 255] = 255
    val[val < 0] = 0
    
    return {'observed_val': val, 'expected_val': expected_val + mu}


test_case = f_data_generating(x, xm, mask, N)

plt.figure()
plt.scatter(x, test_case['observed_val'], label='Observations')
plt.plot(x, test_case['expected_val'], color='red', linestyle='solid', 
         label='Underlying model')
plt.axvline(x=xm-1, color='red', linestyle='dotted', 
            label='Underlying location of maximum')
plt.ylim(0, 500)
plt.ylabel("Observed value f(x)")
plt.legend(loc='upper right')
plt.show()





def f_model(x, xm, sigma, y_ambient, y_scaling):
    s2 = sigma**2
    val = y_scaling * np.exp(-((x-xm)**2)/s2) + y_ambient
    return val

def fit_criterion(pars, x, y_observed):
    #y_model = f_model(x, pars[0], pars[1], pars[2], pars[3])
    y_model = f_model(x, *pars)
    residues = (y_model - y_observed)
    losses = residues**2

    #losses[y_observed > 254] = 0
    empirical_risk = np.nansum(losses)

    return empirical_risk



observed_val = test_case['observed_val']
#observed_val[observed_val == 255] = np.nan

init_pars = [50, 5, 5, 250]
best_crit = np.inf
pars = np.nan
for i in range(100):
    init_pars = [np.random.normal(50, 10),
                 np.random.normal(5, 1),
                 np.random.normal(5, 1),
                 np.random.normal(250, 100)]
    res = minimize(fit_criterion, init_pars, 
                   args=(x, observed_val),
                   method='L-BFGS-B')
    
    if res['success']:
        if res['fun'] < best_crit:
            print(best_crit)
            best_crit = res['fun']
            pars = res['x']
            
#pars = res['x']

y_fit = f_model(x, pars[0], pars[1], pars[2], pars[3])


plt.figure()
plt.scatter(x, test_case['observed_val'], label='Observations')
plt.plot(x, test_case['expected_val'], color='red', linestyle='solid', 
         label='Underlying model')
plt.axvline(x=xm-1, color='red', linestyle='dotted', 
            label='True maximum')
plt.ylim(0, 500)
plt.ylabel("Observed value f(x)")

plt.plot(x, y_fit, color='blue', label = 'Model')
plt.axvline(x=x[np.argmax(y_fit)], color='blue', linestyle='dotted',
            label = 'Found maximum')
plt.legend(loc='upper right')
plt.show()

