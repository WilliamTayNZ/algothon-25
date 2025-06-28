import numpy as np

nInst = 50
currentPos = np.zeros(nInst)

def getMyPosition(prcSoFar):
    global currentPos
    
    try:
        (n, t) = prcSoFar.shape
        if t < 30:
            return np.zeros(n)

        price_today = prcSoFar[:, -1]
        
        # Ensure we have valid price data
        if np.any(price_today <= 0) or np.any(np.isnan(price_today)) or np.any(np.isinf(price_today)):
            return np.zeros(n)

        # === Enhanced Momentum (multiple timeframes) ===
        try:
            momentum_10 = prcSoFar[:, -1] / prcSoFar[:, -11] - 1
            momentum_20 = prcSoFar[:, -1] / prcSoFar[:, -21] - 1
            momentum_30 = prcSoFar[:, -1] / prcSoFar[:, -31] - 1
            
            # Clean any invalid values
            momentum_10 = np.nan_to_num(momentum_10, nan=0.0, posinf=0.0, neginf=0.0)
            momentum_20 = np.nan_to_num(momentum_20, nan=0.0, posinf=0.0, neginf=0.0)
            momentum_30 = np.nan_to_num(momentum_30, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Weighted momentum (more weight to shorter-term)
            momentum = 0.5 * momentum_10 + 0.3 * momentum_20 + 0.2 * momentum_30
        except:
            momentum = np.zeros(n)

        # === Enhanced Mean Reversion (multiple SMAs) ===
        try:
            sma5 = np.mean(prcSoFar[:, -6:-1], axis=1)
            sma10 = np.mean(prcSoFar[:, -11:-1], axis=1)
            
            # Clean any invalid values
            sma5 = np.nan_to_num(sma5, nan=price_today, posinf=price_today, neginf=price_today)
            sma10 = np.nan_to_num(sma10, nan=price_today, posinf=price_today, neginf=price_today)
            
            deviation_5 = (sma5 - price_today) / (sma5 + 1e-6)
            deviation_10 = (sma10 - price_today) / (sma10 + 1e-6)
            
            # Clean deviations
            deviation_5 = np.nan_to_num(deviation_5, nan=0.0, posinf=0.0, neginf=0.0)
            deviation_10 = np.nan_to_num(deviation_10, nan=0.0, posinf=0.0, neginf=0.0)
            
            # Weighted mean reversion
            mean_reversion = 0.7 * deviation_5 + 0.3 * deviation_10
        except:
            mean_reversion = np.zeros(n)

        # === Dynamic Weight Adjustment ===
        try:
            # Adjust weights based on market volatility
            log_returns = np.diff(np.log(prcSoFar[:, -21:]), axis=1)
            log_returns = np.nan_to_num(log_returns, nan=0.0, posinf=0.0, neginf=0.0)
            market_vol = np.mean(np.std(log_returns, axis=1))
            
            # Higher momentum weight in trending markets, higher mean reversion in choppy markets
            if market_vol > 0.02:  # High volatility - favor mean reversion
                momentum_weight = 0.4
                mean_reversion_weight = 0.6
            else:  # Low volatility - favor momentum
                momentum_weight = 0.7
                mean_reversion_weight = 0.3
        except:
            momentum_weight = 0.6
            mean_reversion_weight = 0.4

        # === Enhanced Hybrid Signal ===
        hybrid_signal = momentum_weight * momentum + mean_reversion_weight * mean_reversion
        hybrid_signal = np.nan_to_num(hybrid_signal, nan=0.0, posinf=0.0, neginf=0.0)

        # === Volatility-adjusted scoring with minimum volatility filter ===
        try:
            vol = np.std(log_returns, axis=1) + 1e-6
            vol = np.nan_to_num(vol, nan=1e-6, posinf=1e-6, neginf=1e-6)
            
            min_vol_threshold = np.percentile(vol, 10)  # Avoid extremely low vol instruments
            max_vol_threshold = np.percentile(vol, 85)  # Avoid top 15% most volatile
            
            valid_instruments = (vol >= min_vol_threshold) & (vol <= max_vol_threshold)
            
            signal_score = hybrid_signal / vol
            signal_score[~valid_instruments] = 0  # Zero out invalid instruments
            signal_score = np.nan_to_num(signal_score, nan=0.0, posinf=0.0, neginf=0.0)
        except:
            signal_score = np.zeros(n)

        # === Enhanced Signal Filtering ===
        try:
            # Only trade if signal is strong enough
            signal_std = np.std(signal_score)
            if signal_std > 0:
                signal_threshold = signal_std * 0.5
                strong_signals = np.abs(signal_score) > signal_threshold
            else:
                strong_signals = np.zeros(n, dtype=bool)
        except:
            strong_signals = np.zeros(n, dtype=bool)

        # === Dynamic Position Sizing ===
        try:
            ranked = np.argsort(signal_score)
            top_long = ranked[-8:]
            top_short = ranked[:8]
            
            # Filter for strong signals only
            top_long = [i for i in top_long if i < n and strong_signals[i] and signal_score[i] > 0]
            top_short = [i for i in top_short if i < n and strong_signals[i] and signal_score[i] < 0]
        except:
            top_long = []
            top_short = []

        # === Enhanced Risk-Adjusted Allocation ===
        newPos = np.zeros(n)
        
        try:
            if len(top_long) > 0:
                long_scores = signal_score[top_long]
                long_sum = np.sum(np.abs(long_scores))
                if long_sum > 0:
                    long_weights = long_scores / long_sum
                    
                    # Dynamic budget based on signal strength
                    total_long_strength = long_sum
                    base_budget = 12000
                    dynamic_budget = min(base_budget, base_budget * (total_long_strength / 0.1))
                    
                    for i, w in zip(top_long, long_weights):
                        if i < n and price_today[i] > 0:
                            exposure = abs(w * dynamic_budget)
                            newPos[i] = int(exposure / price_today[i])

            if len(top_short) > 0:
                short_scores = signal_score[top_short]
                short_sum = np.sum(np.abs(short_scores))
                if short_sum > 0:
                    short_weights = short_scores / short_sum
                    
                    # Dynamic budget based on signal strength
                    total_short_strength = short_sum
                    base_budget = 12000
                    dynamic_budget = min(base_budget, base_budget * (total_short_strength / 0.1))
                    
                    for i, w in zip(top_short, short_weights):
                        if i < n and price_today[i] > 0:
                            exposure = abs(w * dynamic_budget)
                            newPos[i] = -int(exposure / price_today[i])
        except:
            pass

        # === Position Limits ===
        try:
            max_position_value = 15000
            for i in range(n):
                if i < len(price_today) and price_today[i] > 0:
                    if abs(newPos[i] * price_today[i]) > max_position_value:
                        newPos[i] = int(max_position_value / price_today[i]) * np.sign(newPos[i])
        except:
            pass

        # === Volatility Targeting Overlay (with comprehensive error handling) ===
        try:
            # Only calculate if we have positions
            if np.sum(np.abs(newPos)) > 0:
                portfolio_weights = newPos * price_today / (np.sum(np.abs(newPos * price_today)) + 1e-6)
                portfolio_weights = np.nan_to_num(portfolio_weights, nan=0.0, posinf=0.0, neginf=0.0)
                
                # Ensure we have enough data for covariance
                if t >= 21:
                    returns_data = np.diff(np.log(prcSoFar[:, -21:]), axis=1)
                    # Remove any NaN or inf values
                    returns_data = np.nan_to_num(returns_data, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    # Calculate covariance matrix safely
                    if returns_data.shape[0] > 0 and returns_data.shape[1] > 0:
                        cov = np.cov(returns_data)
                        # Ensure covariance matrix is valid
                        if cov.size > 0 and not np.any(np.isnan(cov)) and not np.any(np.isinf(cov)):
                            # Calculate portfolio volatility
                            port_vol = np.sqrt(np.dot(portfolio_weights, np.dot(cov, portfolio_weights))) * np.sqrt(252)
                            target_vol = 0.03
                            
                            # Only scale if portfolio volatility is positive and reasonable
                            if port_vol > 0 and port_vol < 1.0:  # Sanity check
                                scale = target_vol / port_vol
                                # Limit scaling to reasonable bounds
                                scale = np.clip(scale, 0.1, 10.0)
                                newPos = (newPos * scale).astype(int)
        except:
            # If any error occurs, keep original positions
            pass

        # Final safety check
        newPos = np.nan_to_num(newPos, nan=0.0, posinf=0.0, neginf=0.0)
        newPos = newPos.astype(int)
        
        currentPos = newPos
        return currentPos
        
    except Exception as e:
        # If anything goes wrong, return zeros
        return np.zeros(n)