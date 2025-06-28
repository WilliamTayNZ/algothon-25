import numpy as np

nInst = 50
currentPos = np.zeros(nInst)
# Track rolling P&L for dynamic sizing
rollingPL = np.zeros(nInst)
# Track entry price for stop-loss/profit-taking
entryPrice = np.zeros(nInst)

# Parameters
STOP_LOSS_PCT = 0.04  # 4% adverse move
TAKE_PROFIT_PCT = 0.06  # 6% favorable move
ROLLING_PL_WINDOW = 20

# For dynamic thresholding
min_trades = 2
max_trades = 6


def getMyPosition(prcSoFar):
    global currentPos, rollingPL, entryPrice
    (n, t) = prcSoFar.shape
    if t < 40:
        return np.zeros(n)

    # === Market regime filter ===
    # Only trade mean reversion if the average price index is not trending strongly
    market_avg = np.mean(prcSoFar, axis=0)
    x = np.arange(30)
    y = market_avg[-30:]
    market_slope = np.polyfit(x, y, 1)[0] / (np.mean(y) + 1e-6)
    if abs(market_slope) > 0.01:  # If market is trending >1% per day, don't trade
        return np.zeros(n)

    # === Calculate mean reversion signals ===
    short_sma = np.mean(prcSoFar[:, -11:-1], axis=1)
    medium_sma = np.mean(prcSoFar[:, -21:-1], axis=1)
    long_sma = np.mean(prcSoFar[:, -41:-1], axis=1)
    price_today = prcSoFar[:, -1]

    returns = np.diff(np.log(prcSoFar[:, -21:]), axis=1)
    volatility = np.std(returns, axis=1) + 1e-6

    short_dev = (short_sma - price_today) / price_today
    medium_dev = (medium_sma - price_today) / price_today
    long_dev = (long_sma - price_today) / price_today
    short_zscore = short_dev / volatility
    medium_zscore = medium_dev / volatility
    long_zscore = long_dev / volatility

    # Cross-sectional z-score
    cross_sectional = price_today - np.mean(price_today)
    cross_sectional_z = (cross_sectional - np.mean(cross_sectional)) / (np.std(cross_sectional) + 1e-6)

    # Combined signal
    combined_zscore = 0.15 * short_zscore + 0.35 * medium_zscore + 0.35 * long_zscore + 0.15 * cross_sectional_z

    # Trend filter per instrument
    trend_strength = np.zeros(n)
    for i in range(n):
        x = np.arange(20)
        y = prcSoFar[i, -20:]
        slope = np.polyfit(x, y, 1)[0]
        trend_strength[i] = slope / (price_today[i] + 1e-6)
    trend_threshold = 0.03
    trend_filter = np.abs(trend_strength) < trend_threshold

    # Final signal
    final_signal = combined_zscore * trend_filter

    # === Adaptive thresholding ===
    # Try to get between min_trades and max_trades per day
    signal_threshold = 1.0
    num_trades = np.sum((final_signal > signal_threshold) | (final_signal < -signal_threshold))
    if num_trades < min_trades:
        signal_threshold = 0.8
    elif num_trades > max_trades:
        signal_threshold = 1.2

    long_candidates = final_signal > signal_threshold
    short_candidates = final_signal < -signal_threshold

    # === Dynamic position sizing based on rolling P&L ===
    # Use a rolling window of PL to adjust risk per instrument
    # If recent PL is negative, reduce size; if positive, increase
    base_size = 8000
    pos_size = np.ones(n) * base_size
    for i in range(n):
        # Calculate rolling PL for last ROLLING_PL_WINDOW days
        if t > ROLLING_PL_WINDOW:
            pl = prcSoFar[i, -ROLLING_PL_WINDOW:] - prcSoFar[i, -ROLLING_PL_WINDOW-1:-1]
            rollingPL[i] = np.sum(pl)
            if rollingPL[i] < 0:
                pos_size[i] = base_size * 0.7
            elif rollingPL[i] > 0:
                pos_size[i] = base_size * 1.3
            else:
                pos_size[i] = base_size
        else:
            rollingPL[i] = 0
            pos_size[i] = base_size

    # === Position selection ===
    max_positions = 6
    newPos = np.zeros(n)

    # Longs
    if np.sum(long_candidates) > 0:
        long_indices = np.where(long_candidates)[0]
        long_strengths = final_signal[long_indices]
        sorted_long = sorted(zip(long_indices, long_strengths), key=lambda x: x[1], reverse=True)
        for i, (idx, strength) in enumerate(sorted_long[:max_positions//2]):
            vol_factor = 1 / (volatility[idx] + 1e-6)
            vol_factor = np.clip(vol_factor, 0.5, 2.5)
            size = pos_size[idx] / price_today[idx]
            size = int(size * vol_factor * min(strength / 1.5, 1.2))
            newPos[idx] = min(size, int(9500 / price_today[idx]))
            entryPrice[idx] = price_today[idx]  # Set entry price for stop-loss

    # Shorts
    if np.sum(short_candidates) > 0:
        short_indices = np.where(short_candidates)[0]
        short_strengths = final_signal[short_indices]
        sorted_short = sorted(zip(short_indices, short_strengths), key=lambda x: x[1])
        for i, (idx, strength) in enumerate(sorted_short[:max_positions//2]):
            vol_factor = 1 / (volatility[idx] + 1e-6)
            vol_factor = np.clip(vol_factor, 0.5, 2.5)
            size = pos_size[idx] / price_today[idx]
            size = int(size * vol_factor * min(abs(strength) / 1.5, 1.2))
            newPos[idx] = -min(size, int(9500 / price_today[idx]))
            entryPrice[idx] = price_today[idx]

    # === Stop-loss and profit-taking logic ===
    for i in range(n):
        if currentPos[i] > 0:
            # If price drops more than STOP_LOSS_PCT from entry, close
            if price_today[i] < entryPrice[i] * (1 - STOP_LOSS_PCT):
                newPos[i] = 0
            # If price rises more than TAKE_PROFIT_PCT from entry, close
            if price_today[i] > entryPrice[i] * (1 + TAKE_PROFIT_PCT):
                newPos[i] = 0
        elif currentPos[i] < 0:
            # If price rises more than STOP_LOSS_PCT from entry, close
            if price_today[i] > entryPrice[i] * (1 + STOP_LOSS_PCT):
                newPos[i] = 0
            # If price drops more than TAKE_PROFIT_PCT from entry, close
            if price_today[i] < entryPrice[i] * (1 - TAKE_PROFIT_PCT):
                newPos[i] = 0

    # === Additional safety: cap to top 3 each side ===
    long_count = np.sum(newPos > 0)
    short_count = np.sum(newPos < 0)
    if long_count > 3:
        long_positions = [(i, final_signal[i]) for i in range(n) if newPos[i] > 0]
        long_positions.sort(key=lambda x: x[1], reverse=True)
        for i in range(3, len(long_positions)):
            newPos[long_positions[i][0]] = 0
    if short_count > 3:
        short_positions = [(i, final_signal[i]) for i in range(n) if newPos[i] < 0]
        short_positions.sort(key=lambda x: x[1])
        for i in range(3, len(short_positions)):
            newPos[short_positions[i][0]] = 0

    currentPos = newPos
    return currentPos
