import numpy as np

nInst = 50
currentPos = np.zeros(nInst)

def getMyPosition(prcSoFar):
    global currentPos
    (n, t) = prcSoFar.shape
    if t < 21:
        return np.zeros(n)

    price_today = prcSoFar[:, -1]

    # === Momentum (20-day return) ===
    momentum = prcSoFar[:, -1] / prcSoFar[:, -21] - 1

    # === Mean reversion: deviation from 5-day SMA ===
    sma5 = np.mean(prcSoFar[:, -6:-1], axis=1)
    deviation = sma5 - price_today

    # === Combine signals ===
    hybrid_signal = 0.6 * momentum + 0.4 * (deviation / (sma5 + 1e-6))

    # === Volatility-adjusted scoring ===
    log_returns = np.diff(np.log(prcSoFar[:, -6:]), axis=1)
    vol = np.std(log_returns, axis=1) + 1e-6
    signal_score = hybrid_signal / vol

    # === Rank signals ===
    ranked = np.argsort(signal_score)
    top_long = ranked[-10:]
    top_short = ranked[:10]

    # === Allocate positions (risk-adjusted) ===
    newPos = np.zeros(n)
    max_budget = 10000  # hard limit
    long_weights = signal_score[top_long] / np.sum(np.abs(signal_score[top_long]))
    short_weights = signal_score[top_short] / np.sum(np.abs(signal_score[top_short]))

    for i, w in zip(top_long, long_weights):
        exposure = min(max_budget, abs(w * max_budget))
        newPos[i] = int(exposure / price_today[i])

    for i, w in zip(top_short, short_weights):
        exposure = min(max_budget, abs(w * max_budget))
        newPos[i] = -int(exposure / price_today[i])

    currentPos = newPos
    return currentPos
