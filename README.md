# API Integration, Natural Language Processing, Data Cleaning, Calculations and Updating SQL Database for Machine Learning

- The universe uses around 6,000 stocks both listed and delisted since approximately 2007.
- The calculations in this file use the quarterly financial statements and daily prices from the SimFin API to create daily ratios for the stocks. The weighted averages are used to calculate the universe, sector, and industry ratios to compare the stock against its peers within each category.
- Natural Language Processing uses news articles from GNews API which are analyzed by the IBM Watson Tone Analyzer to analyze the sentiment of the articles.
- Functions created in separate Python file and called into the Jupyter Lab Notebook.
- SQL commands are manipulated in Jupyter Lab.
- SQL database is updated based on missing dates.
- Code has checkpoints because the dataset is large being manipulated in the RAM.
- Recoded to take about 45min to run from about 2.5 hours.
- Option to create new database from API dataframe or trim the data to update the database to run faster.
- Code also has a “percentage of completion” measure for the computationally intensive portions of the program.

Ratios and weighted ratios for the stocks, universe sectors and industries are:
- Piotroski F-Score
- P/E Ratio
- Total Shareholder Yield
- Enterprise Yields
- Coverage Ratios
- Operating Yield
- And many others
- Along with Technical Analysis such as volume


# Artificial Intelligence - Machine Learning
# Long Short-Term Memory (LSTM) Recurrent Neural Network (RNN) Unsupervised Deep Learning Model

- The LSTM model learns long term dependencies and the current input in variables for time series data using a neural network comprised of layers and neurons.
- scikit-learn and TenserFlow Keras are used to train, test, and evaluate the model.
- The ratios calculated in the API integration are used as features (minus the target column [Market Cap]) to predict the Market Cap in the LSTM Model.
- A total of 82 features used to predict the Market Cap with a sliding window.
- Data is shifted to avoid look-ahead bias.
- A callback is used to finish the model if the validation loss is no longer decreasing.
- A dashboard is created to show the predictions vs the actual values along with the test and validation loss to check for overfitting.



# Dashboard
“Model Results” (Figure 1) shows the actual market cap vs the predicted market cap.
“Model Loss” (Figure 2) shows if the model is overfit.

![image](https://user-images.githubusercontent.com/71287557/116837191-5df9de80-ab97-11eb-9e61-4cbb530efa16.png)
Figure 1

![image](https://user-images.githubusercontent.com/71287557/116837212-710cae80-ab97-11eb-91eb-63c729aa8185.png)
Figure 2


# Disclaimer
- This is for demonstration purposes only to showcase the coding methodologies. There are some inherent flaws in the data and not meant to be used for any real analysis.
- No filter for significant / relevant features creates a flaw as the model will make a connection with data and find a value even if it does not make sense.
- Early data collected does not calculate market cap correctly forcing ratios to be wrong. News articles collected do not go back far, so there is not a lot of testing and training data (about 2 years of good data that must be split between test and train).
- Only AAPL is analyzed as that is the only stock the news was collected for because of API limitations.
