import pandas as pd

def calculate_vwap(data, closeCol='close', volumeCol='volume'):
    """
    the Volume-Weighted Average Price (VWAP) is a trading indicator that calculates the average price a
    stock has traded at throughout the day, weighted by the trading volume at each price level.

    Parameters
    ----------
    data
    closeCol
    volumeCol
    decimals

    Returns
    -------

    """

    cumulative_volume_price = (data[volumeCol] * data[closeCol]).cumsum()
    cumulative_volume = data[volumeCol].cumsum()
    vwap = cumulative_volume_price / cumulative_volume
    return vwap


def calculate_stochastic_oscillator(data, period=14, smoothing=3):
    """
     Traders often use the crossing of the %K and %D lines or the position of these lines in relation to
     overbought (typically above 80) and oversold (typically below 20) levels as potential buy or sell signals.

    Parameters
    ----------
    data
    period
    smoothing
    decimals

    Returns
    -------

    """

    # Calculate the lowest low and highest high over the specified period
    lowest_low = data.rolling(window=period).min()
    highest_high = data.rolling(window=period).max()

    # Calculate the %K
    k_percent = 100 * ((data - lowest_low) / (highest_high - lowest_low))

    # Calculate the %D (smoothing)
    d_percent = k_percent.rolling(window=smoothing).mean()

    return k_percent, d_percent


def calculate_rsi(data, period=14):
    """
    Relative Strength Index (RSI): RSI measures the speed and change of price movements to determine
    overbought or oversold conditions. It is often used to identify potential reversal points in the market.

    provides insights into whether a stock is overbought or oversold and can indicate potential trend reversals

    RSI value above 70 is considered overbought, suggesting that the stock may be due for a pullback.
    Conversely, an RSI value below 30 is considered oversold,
    Parameters
    ----------
    data
    period
    decimals

    Returns
    -------

    """

    # Calculate price changes
    delta = data.diff().dropna()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)

    # Calculate average gains and losses
    avg_gain = gains.rolling(window=period).mean()
    avg_loss = losses.rolling(window=period).mean()

    # Calculate relative strength (RS)
    rs = avg_gain / avg_loss

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_cmo(data, period):
    """
    the technical indicator you're referring to is likely the "Chande Momentum Oscillator" (CMO). The Chande Momentum
    Oscillator is a technical analysis tool developed by Tushar Chande, a prominent technical analyst. It is used to
    measure the momentum of a financial instrument, such as a stock or a market index.

    The CMO calculates the difference between the sum of all recent gains and the sum of all recent losses over a specified
    period. The result is then normalized within a range of -100 to +100 to provide an oscillating indicator.
    The CMO is designed to identify overbought and oversold conditions, as well as potential trend reversals.

    interpretation of the CMO is as follows:

    Values above +50 are typically considered overbought, indicating a potential downside reversal or a correction in price.
    Values below -50 are often considered oversold, suggesting a possible upside reversal or a bounce back in price.
    Values near zero indicate a lack of momentum or a consolidating market.

    Parameters
    ----------
    data
    period

    Returns
    -------

    """
    gains = []
    losses = []

    for i in range(1, len(data)):
        diff = data[i] - data[i - 1]
        if diff > 0:
            gains.append(diff)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(diff))

    sma_gains = sum(gains[:period]) / period
    sma_losses = sum(losses[:period]) / period

    cmo_values = []

    for i in range(period, len(data)):
        sma_gains = (sma_gains * (period - 1) + gains[i]) / period
        sma_losses = (sma_losses * (period - 1) + losses[i]) / period
        cmo = 100 * ((sma_gains - sma_losses) / (sma_gains + sma_losses))
        cmo_values.append(cmo)

    return cmo_values

def calculate_technical_indicators(data, indicators, closeColumn="close", volumeCol='volume', decimals=2):
    """
    you can define which indicators you want to calculate by passing a list of strings to the indicators parameter.
    besides  the technical indicator you can also calculater the percentage change of the close price (PCTC).

    The following indicators are currently supported:
        Relative Strength Index (RSI)
        Stochastic Oscillator (SO)
        Volume-Weighted Average Price (VWAP)

    You can specify the period of the indicator by adding an underscore and the period to the indicator shortcut.
    For example, to calculate the RSI with a period of 14, you would pass "RSI_14" to the indicators parameter.

    WARNING: when calculating RSI and S0, the first n values will be NaN, where n is the period of the indicator

    Parameters
    ----------
    data
    indicators: list
        ex.: ["RSI_14","RSI_2", "SO_4","VWAP","PCTC_4"]
    closeColumn: str
        defines the column name of the close price
    volumeCol: str
        defines the column name of the volume
    decimals

    Returns
    -------

    """

    try:
        for indicator in indicators:
            if "RSI" in indicator:
                arguments = indicator.split("_")
                data["RSI_" + arguments[1]] = calculate_rsi(data[closeColumn], int(arguments[1]))
            elif "SO" in indicator:
                arguments = indicator.split("_")
                data["SOk_" + arguments[1]], data["SOd_" + arguments[1]] = calculate_stochastic_oscillator(
                    data[closeColumn], int(arguments[1]), decimals)
            elif "VWAP" in indicator:
                VW = calculate_vwap(data, closeColumn, volumeCol)
                data["VWAP"] =VW
            elif "PCTC" in indicator:
                arguments = indicator.split("_")
                data["PCTC_" + arguments[1]] = data[closeColumn].pct_change(int(arguments[1]))

    except ValueError as e:
        print("WARNING:", e)
    except Exception as e:
        print("WARNING:", e)
    return data

