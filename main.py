#This code fetches historical price data from Yahoo Finance for a list of funds, 
#calculates daily returns, computes pairwise correlations between the returns of all pairs of funds,
# and calculates and filters drawdowns for each fund.
# A drawdown is a peak-to-trough decline during a specific period for an investment, trading account, or fund.
# It also plots the rolling correlation between two funds selected using an interactive widget and the cumulative distribution function of all pairwise correlations.

#We first import the required libraries.
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# List of ticker symbols for the funds we're interested in.
funds = ["IVV", "IYW", "IYF", "IYZ", "IYH", "IYK", "IYE", "IYJ", "IDU", "IYM", "IYC", "IYR"]

# We initialize an empty DataFrame to store our data.
df = pd.DataFrame()

# We iterate over each fund in our list of funds.
for fund in funds:
    # We download the historical price data for the fund from Yahoo Finance.
    data = yf.download(fund, start="2000-06-12", end="2023-06-06")
    
    # We check if our DataFrame is empty.
    if df.empty:
        # If it is, we set it equal to the 'Close' prices from the data we just downloaded.
        df = data[['Close']]
        
        # We rename the column in the DataFrame to be the name of the fund.
        df.rename(columns = {'Close': fund}, inplace = True)
    else:
        # If our DataFrame is not empty, we add a new column to it with the 'Close' prices from the data we just downloaded.
        df[fund] = data["Close"]

# We remove any rows from our DataFrame that have missing data.
df = df.dropna()

# We calculate the daily returns for each fund by using the 'pct_change' method.
returns = df.pct_change()

# We remove the first row from our DataFrame of returns, as it will be NaN (because there's no previous day to calculate a return from).
returns = returns.dropna()

# We calculate the pairwise correlation of the returns of all the funds.
correlations = returns.corr()

# We define a function to compute drawdowns.
def compute_drawdowns(data_series):
    """
    Function to compute drawdowns based on a pandas Series
    A drawdown is a peak-to-trough decline during a specific period for an investment, trading account, or fund.
    """
    # We first calculate the cumulative returns for the data series.
    cumulative_returns = (1 + data_series).cumprod()
    
    # We then calculate the running maximum of these cumulative returns.
    running_max = cumulative_returns.cummax()
    
    # We calculate drawdowns by subtracting the running max from the cumulative returns and then dividing by the running max.
    drawdowns = (cumulative_returns - running_max) / running_max
    
    return drawdowns

# We calculate the drawdowns for the fund 'IVV'.
drawdowns_IVV = compute_drawdowns(returns["IVV"])

# We filter the drawdowns for 'IVV' to only include those that are greater than 10%.
drawdowns_IVV = drawdowns_IVV[drawdowns_IVV <= -0.10]

# We define a function to compute and plot the rolling correlation between two funds.
def plot_rolling_corr(fund_a, fund_b, window):
    # We calculate the rolling correlation between the two funds over a specified window of days.
    rolling_corr = returns[[fund_a, fund_b]].rolling(window).corr().unstack()[fund_a][fund_b]
    
    # We plot this rolling correlation.
    rolling_corr.plot(figsize=(14,7))
    
    # We add a title to the plot.
    plt.title('Rolling Correlation Between {} and {}'.format(fund_a, fund_b))
    
    # We display the plot.
    plt.show()

# We create an interactive widget that lets us select two funds and a window size and then plots the rolling correlation between the two funds over that window.
plot_rolling_corr('IVV', 'IYF', 252)
# We set a window size of 30 days.
window = 30

# We calculate the rolling correlations between all pairs of funds over this window.
all_pairwise_corrs = returns.rolling(window).corr().unstack().values.flatten()

# We remove any NaN values from our array of correlations.
all_pairwise_corrs = all_pairwise_corrs[~np.isnan(all_pairwise_corrs)]

# We plot the cumulative distribution function (CDF) of these correlations.


# The Cumulative Distribution Function (CDF) of pairwise correlations is a plot that shows the probability that a randomly selected correlation from the set of all pairwise correlations is less than or equal to a particular value.

# In the context of this code, where it calculates the CDF of pairwise correlations between the returns of a set of funds, it is used to visualize the distribution of these correlations.

# Here's how to interpret it:

# The x-axis represents the correlation value.
# The y-axis represents the cumulative probability.
# For any given x (correlation) value, the corresponding y value represents the probability that a randomly chosen pairwise correlation is less than or equal to that x value.

# For example, if you have a point (0.5, 0.7) on the CDF, this means that 70% of all the pairwise correlations are less than or equal to 0.5.


# This kind of analysis is useful for understanding the overall correlation structure among the funds. For instance, if the CDF rises quickly to 1, it means that most of the correlations are concentrated in a smaller range of values. If it rises slowly, it means that the correlations are more spread out.


# 
# Knowing the distribution of pairwise correlations in a portfolio of assets can provide important insights for investment decision making.

# Risk Management: Correlation measures the degree to which two securities move in relation to each other. If all the assets in a portfolio are highly correlated, they will likely move in the same direction under similar market conditions. This could lead to substantial losses if those conditions are unfavorable. Therefore, understanding the distribution of correlations can help in assessing portfolio risk.

# Diversification: In order to reduce risk, investors often look for assets that are not perfectly correlated. The idea is that when some assets are down, others might be up, smoothing out the overall portfolio returns. The CDF of pairwise correlations can give a quick sense of how diversified the portfolio might be. For example, if 70% of the correlations are less than 0.5, it suggests that there is a good level of diversification in the portfolio.

# Investment Strategy: The correlation structure can influence investment strategies. For instance, pairs trading strategies look for pairs of stocks that are highly correlated. If the correlations shift over time, the strategy may need to be adjusted.

# By plotting the CDF of pairwise correlations, you can get a sense of the overall correlation structure of the portfolio, which can inform these and other aspects of investment decision making.




sns.kdeplot(all_pairwise_corrs, cumulative=True)
plt.title('CDF of Pairwise Correlations')
plt.show()


# We calculate the drawdowns for all the funds.
drawdowns = {}
for fund in funds:
    drawdowns[fund] = compute_drawdowns(returns[fund])

# We filter the drawdowns for each fund to only include those that are greater than 10% and compute some metrics on these drawdowns.
# The median drawdown is the middle point of all drawdowns when they are listed in numerical order. It's a statistical measure that aims to express a 'typical' drawdown value.
# This median value is less sensitive to extreme values (either very high or very low) than the average (mean) drawdown, making it a more robust measure for skewed distributions, which are common in financial returns.
filtered_drawdowns = {}
for fund in funds:
    filtered_drawdowns[fund] = drawdowns[fund][drawdowns[fund] <= -0.10]
    print("Metrics for ", fund) # We print the name of the fund.
    print("Peak to trough days: ", filtered_drawdowns[fund].idxmin() - filtered_drawdowns[fund].index[0]) # We print the peak to trough days for the fund.
    print("Trough to recovery days: ", filtered_drawdowns[fund].index[-1] - filtered_drawdowns[fund].idxmin()) # We print the trough to recovery days for the fund.
    print("Max drawdown: ", filtered_drawdowns[fund].min()) # We print the max drawdown for the fund. Important because its a  measure of an asset's largest price drop from a peak to a trough. 
    print("Median drawdown: ", filtered_drawdowns[fund].median()) # We print the median drawdown for the fund. 
    print("Mean drawdown: ", filtered_drawdowns[fund].mean())  
    print("--------------------------------------------------")
    
#  The mean drawdown is the average of all drawdowns in a given data set or time period. Drawdown, in the context of finance, refers to the decline from a peak in the value of an investment or portfolio. It's typically expressed as a percentage and is used to measure the risk or volatility of the investment.

# To calculate the mean drawdown, you would:

# Identify all the peak-to-trough declines (drawdowns) in the period you're examining.
# Calculate the percentage decline for each drawdown.
# Sum all these percentages.
# Divide by the number of drawdowns to get the mean (average).
# The mean drawdown can provide a sense of the typical level of loss from peak to trough that the investment or portfolio experienced over the studied period. However, it's important to note that the mean can be significantly influenced by extreme values. Therefore, when the distribution of drawdowns is skewed or has outliers, the median drawdown might be a more representative measure.



# using the OpenAI API to generate text
import openai

# Set your OpenAI GPT-3 API key
openai.api_key = 'OpenAI API KEY'

# Convert your correlations data to a readable format for GPT-3
correlations_str = correlations.to_string()

prompt = f"""
Based on the following correlation matrix of returns for various ETFs:

{correlations_str}

Provide a detailed analysis in the style of a financial news article, considering the following points:

1. Identify the pairs of ETFs with the highest and lowest correlations. What does this mean for the relationship between these funds, and how might these correlations inform future price movements?
2. How might an investor use these correlation figures to inform decisions about portfolio diversification? Consider both higher correlated pairs and lower correlated pairs in your response.
3. Discuss the potential impact of external factors, such as economic changes or shifts in market sentiment, on these correlations. How might bullish or bearish market conditions affect the relationships between these ETFs?
4. Discuss the limitations of using historical correlations as indicators of future performance. How reliable might these figures be for making investment decisions?
5. Propose potential investment strategies that take into account these correlation figures. How might an investor leverage these correlations to optimize their portfolio, considering both risk reduction and potential returns?
6. Also, include any other insights or interesting trends you find from the correlation matrix.
"""



# Call the OpenAI API
response = openai.Completion.create(
  engine="text-davinci-003",
  prompt=prompt,
  temperature=0.5,
  max_tokens=1200,
)

#Print the prediction from GPT-3
print('AI Prediction based on correlations: \n')
print(response.choices[0].text.strip())
