
"""actually use dummy regressor here
"""

coefs = list()
for chan, this_psd in zip(lm_prop.T[0, :], psds.swapaxes(0, 1)):
    dmat = np.array([run_index, chan]).T
    lm = LinearRegression().fit(dmat, this_psd)
    coefs.append(lm.coef_)
coefs = np.array(coefs)