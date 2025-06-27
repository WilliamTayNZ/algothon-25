
import numpy as np

##### TODO #########################################
### IMPLEMENT 'getMyPosition' FUNCTION #############
### TO RUN, RUN 'eval.py' ##########################

nInst = 50 # number of instruments (Stocks)
currentPos = np.zeros(nInst) # your current positions in each stock, initialized to 0 for all 50 instruments


def getMyPosition(prcSoFar): # takes as input an NP array of shape (50, t), where t is the number of days so far.
    global currentPos # Ensures we are modifying the global currentPos
    (nins, nt) = prcSoFar.shape
    if (nt < 2):
        return np.zeros(nins)
    lastRet = np.log(prcSoFar[:, -1] / prcSoFar[:, -2]) # Computes log returns from the most recent 2 days for each stock
    lNorm = np.sqrt(lastRet.dot(lastRet)) # calculates L2 norm (Euclidean norm) (length or magnitude) of the return vector — used to normalize
    lastRet /= lNorm # normalizes the return vector to have unit length so large-magnitude moves don't dominate the portfolio.

    # Scales the signal (lastRet) and converts to dollar positions (in units of shares)
    # 5000 * signal / price ≈ number of shares to buy/sell.
    rpos = np.array([int(x) for x in 5000 * lastRet / prcSoFar[:, -1]])

    # Adds the position change to the running total currentPos.
    currentPos = np.array([int(x) for x in currentPos+rpos])
    return currentPos

''' 
Summary:
This strategy reacts only to the last day's return.
It buys if the stock went up, sells if it went down.
Risk is (weakly) controlled by normalizing return vector.

'''