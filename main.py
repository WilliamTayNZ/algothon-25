import numpy as np

nInst = 50
currentPos = np.zeros(nInst)

def getMyPosition(prcSoFar):
    global currentPos
    (n, t) = prcSoFar.shape
    if t < 6:
        return np.zeros(n)

    # === Mean and deviation over last 5 days ===
    sma = np.mean(prcSoFar[:, -6:-1], axis=1)  # 5-day moving average
    price_today = prcSoFar[:, -1]
    deviation = sma - price_today  # positive = underpriced, negative = overpriced

    # === Volatility adjustment (standard deviation of last 5 returns) ===
    returns = np.diff(np.log(prcSoFar[:, -6:]), axis=1)
    vol = np.std(returns, axis=1) + 1e-6

    # === Z-score of deviation ===
    signal = deviation / vol

    # === Rank and select strongest signals ===
    ranked = np.argsort(signal)  # ascending
    long_idx = ranked[-5:]
    short_idx = ranked[:5]

    # === Allocate capital ===
    newPos = np.zeros(n)
    budget = 9500  # stay under $10k cap

    for i in long_idx:
        newPos[i] = int(budget / price_today[i])
    for i in short_idx:
        newPos[i] = -int(budget / price_today[i])

    currentPos = newPos
    return currentPos
