# Processing Sequences using RNNs and CNNs

## RNN
RNNs can work on sequences of arbitrary lengths, rather than on fixed-sized inputs.

RNNs have two main difficulties:
1. **Unstable gradients**, which can be alleviated using varous techniques, including recurrent dropout and recurrent layer normalization.
2. **A (very) limited short-term memory**, which can be extended using LSTM and GRU cells.

- For small sequences, a regular dense network can do the trick.
- For very long sequences, convolutional neural networks can actually work quite well too.

### Recurrent Neurons and Layers
![A recurrent neuron (left) unrolled through time (right)](images/photo_2024-04-15_21-43-37.jpg)

A recurrent neuron (left) unrolled through time (right)

![A layer of recurrent neurons (left) unrolled through time (right)](images/photo_2024-04-15_21-53-25.jpg)

A layer of recurrent neurons (left) unrolled through time (right)

### Memory Cells 
Since the output of a recurrent neuron at time step t is a function of all the inputs from previous time steps, you could say it has a form of memory.

A single recurrent neuron or a layer of recurrent neurons, is a very basic cell, capable of learning only short patterns (typically about 10 steps long, but this varies depending on the task).

h(t) = f(x(t), h(t-1))

In the case of the basic cells, we have discussed so far, the output is just equal to the state.

### Input and Output Sequences
1. sequence-to-sequence network: forecast time series
2. sequence-to-vector network: ignore all outputs except for the last one. feed the network a sequence of words and the network would output a sentiment score
3. vector-to-sequence network: caption for an image
4. encoder-decoder: a sequence-to-vector network (encoder) followed by a vector-to-sequence network (decoder). In translating a sentence works much better than trying to translate on the fly  with a single sequence to sequence RNN. The last words of a squence can affect the first words of the translation.

![Sequence-to-sequence (top left), sequence-to-vector (top right), vector-tosequence
(bottom left), and encoder–decoder (bottom right) networks](images/photo_2024-04-20_13-10-14.jpg)

Sequence-to-sequence (top left), sequence-to-vector (top right), vector-tosequence
(bottom left), and encoder–decoder (bottom right) networks

### Training RNNs
![Backpropagation through time)](images/bptt.jpg)

Backpropagation through time

### Forecasting a Time Series
naive forecasting: simply copying a past value to make our forecast. Naive forecasting is often a great baseline, and it can even be tricky to beat in some cases.

When a time series is correlated with a lagged version of itself, we say that the time series is autocorrelated.

![time series)](images/time-series.jpg)

Time series overlaid with 7-day lagged time series (top), and difference between t and t-7 (bottom)

Differencing is a common technique used to remove trend and seasonality from a time series: it's easier to study a **stationary** time series, meaning one whose statistical properties remain constant over time, without any seasonality or trends. Once you are able to make accurate forecasts on the differenced time series, it's easy to turn them into forecasts for the actual time series by just adding back the past values that were previously subtracted.

### The ARMA Model Family
Autoregressive moving average (ARMA): Coumputes its forecasts using a simple weighted sum of lagged values and corrects these forecasts by adding a moving average.

![images/photo_2024-11-04_18-08-33.jpg](images/photo_2024-11-04_18-08-33.jpg)

The first sum is the autoregressive component. The second sum is the moving average component.

The model assumes the time series is stationary. Using differencing over a single time step will produce an approximation of the derivative of the time series (slope of the series) (eliminate any linear trend).

If the original time series has a quadratic trend, then a single round of differencing will not be enough. Running d consecutive rounds of differencing computes an approximation of the dth order derivative of the time series, so it will eliminate polynomial trends up to degree d. This hyperparameter d is called the order of integration.

Autoregressive integrated moving average (ARIMA) runs d rounds of differencing to make the time series more stationary, after applying ARMA it adds back the terms that were subtracted by differencing.

Seasonal ARIMA (SARIMA): models the time series in the same way as ARIMA, but it additionally models a seasonal component for a given frequency (e.g., weekly).

There are principled approaches to selecting good hyperparameters, based on analyzing the autocorrelation function (ACF) and partial autocorrelation function (PACF_, or minimizing the AIC or BIC metrics to penalize models that use too many parameters.

Since gradient descent expects the instances in the training set to be independent and identically distributed, we must set the argument shuffle=True to shuffle the training windows.

### Forecasting Multivariate Time Series
Using a single model for multiple related tasks often results in better performance than using a separate model for each task, since features learned for one task may be useful for the other tasks, and also because having to perform well across multiple tasks prevents the model from overfitting (it's a form of regularization).
