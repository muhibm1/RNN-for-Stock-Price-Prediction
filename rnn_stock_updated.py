import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as web
import math
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")


class RNN:

    def __init__(self, output_weights=0.001):
        self.input_weight = [1,1]
        self.output_weight = [output_weights, output_weights]
        self.hidden_weights = [0, 0]
        self.scale_up = 1.2
        self.scale_down = .3
        self.history = []
    

    def combine(self, input_value, prev_hidden):
        hidden_value = input_value*self.input_weight[0] +  prev_hidden*self.input_weight[1]
        return hidden_value 
    

    def forward_pass(self, X):
        S = np.zeros((X.shape[0], X.shape[1]+1))
        for k in range(0 , X.shape[1]):
            next_state = self.combine(X[:,k], S[:,k])
            S[:, k+1] = next_state
        return S


    def gradient(self, guess, real, num_samples):
        return (guess - real)/num_samples


    def backward_gradient(self, X, S, grad_out):
        grad_over_time = np.zeros((X.shape[0], X.shape[1] + 1))
        grad_over_time[:, -1] = grad_out

        input_gradient = 0
        hidden_gradient = 0

        for k in range(X.shape[1], 0, -1):
            input_gradient += np.sum(grad_over_time[:, k] * X[:, k-1])
            hidden_gradient += np.sum(grad_over_time[:, k] *S[:, k-1])

            grad_over_time[:, k-1] = grad_over_time[:, k] * self.input_weight[1]
            
        return(input_gradient, hidden_gradient) , grad_over_time
    

    def update_rprop(self, X, Y, W_prev_sign, output_weight, num_samples, verbose):
        S = self.forward_pass(X)
        grad_out = self.gradient(S[:, -1], Y, num_samples)
        loss = np.mean(grad_out)
        if verbose:
            print(f'Loss: {loss}')
        self.history.append(loss)

        W_grads, _ = self.backward_gradient(X, S, grad_out)
        self.hidden_weights = np.sign(W_grads)

        for i, _ in enumerate(self.input_weight):
            if self.hidden_weights[i] == W_prev_sign[i]:
                output_weight[i] *= self.scale_up
            else:
                output_weight[i] *= self.scale_down
            
            self.output_weight = output_weight
    

    def train(self, X, Y, training_steps, num_samples, verbose=True):
        for step in range(training_steps):
            self.update_rprop(X, Y, self.hidden_weights, self.output_weight, num_samples, verbose)

            for i, _ in enumerate(self.input_weight):
                self.input_weight[i] -= self.hidden_weights[i] * self.output_weight[i]


def graph_rnn(rnn, save=True, show=True):
    df = pd.DataFrame({'x': range(0, len(rnn.history)), 'loss': rnn.history})
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)
    ax1.plot('x', 'loss', data=df)
    ax1.set(title = "Epoch vs. Loss", xlabel='Epoch', ylabel='Loss')
    if save:
        fig1.savefig('loss_graph.png', dpi=500)
    if show:
        fig1.show()


def graph_prediction(train, valid, stock, save=True, show=True):
    fig2 = plt.figure() 
    ax1 = fig2.add_subplot(111)
    title = stock + " Prediction"
    ax1.set(title=title, xlabel='Date', ylabel='Close Price USD ($)')
    ax1.plot(train['Close'])
    ax1.plot(valid[['Close', 'Prediction']])
    ax1.legend(['Train', 'Val','Predictions'], loc='upper left')
    if save:
        fig2.savefig('prediction_graph.png', dpi=500)
    if show:
        fig2.show()


def get_data():
    stock = "NFLX"
    start_date = '2012-01-01'
    end_date = '2022-11-29'
    start = pd.to_datetime([start_date]).view(int)[0]//10**9 
    end = pd.to_datetime([end_date]).view(int)[0]//10**9
    url = 'https://query1.finance.yahoo.com/v7/finance/download/' + stock + '?period1=' + str(start) + '&period2=' + str(end) + '&interval=1d&events=history'
    df = pd.read_csv(url)

    return df, stock


def preprocess_data(df):
    df = df.filter(['Close'])
    data = df.values
    num_samples = len(data)
    training_length = int(np.ceil(num_samples * .8))

    sc = MinMaxScaler(feature_range=(0,1))
    data_scaled = sc.fit_transform(data)

    training_data = data_scaled[0: training_length, :]
    X_train = []
    y_train = []
    T = 10 #Number of days to go back

    for i in range(T, len(training_data)):
        X_train.append(training_data[i - T: i, 0])
        y_train.append(training_data[i, 0])

    test_data = data_scaled[training_length - T:, :]
    X_test = []
    y_test = data_scaled[training_length:, :]

    for i in range(T, len(test_data)):
        X_test.append(test_data[i-T: i, 0])

    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    y_test = np.array(y_test)

    data = (X_train, y_train, X_test, y_test)

    return data, num_samples, training_length, sc


def main():
    df, stock = get_data()
    data, num_samples, training_length, sc = preprocess_data(df)
    X_train, y_train, X_test, y_test = data
    log = pd.DataFrame(columns=['error', 'epochs', 'output_weights', 'input_weights'])

    best_rmse = float('inf')
    best_model = None
    best_epochs = 0
    for i in range(10, 100, 10):
        for j in [0.001, 0.005, 0.01, 0.05, 0.1, 0.5]:

            input_w = []
            error = []
            epochs = []
            output_w = []
            
            rnn = RNN(output_weights=j)
            epoch = i
            rnn.train(X_train, y_train, epoch, num_samples, verbose=False) # verbose=True to print the loss to terminal

            y = rnn.forward_pass(X_test)[:, -1]
            y = np.array(y)
            y = sc.inverse_transform(y.reshape(-1, 1))
            real = sc.inverse_transform(y_test)
            rmse = np.sqrt(np.mean(y - real)**2)

            error.append(rmse)
            epochs.append(epochs)
            input_w.append(rnn.input_weight)
            output_w.append(rnn.output_weight)
            log['error'] = error
            log['epochs'] = epochs
            log['output_weights'] = output_w
            log['input_weights'] = input_w

            log.to_csv('log.csv', mode='a', index=False, header=True)

            train = df[:training_length]
            valid = df[training_length:]
            valid['Prediction'] = y

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = (rnn, train, valid)
    rnn, train, valid = best_model
    print(f'Best RNN Model used weights: {rnn.output_weight} and epochs: {best_epochs} resulted in rmse: {best_rmse}')
    graph_rnn(rnn, save=False)
    graph_prediction(train, valid, stock, save=False)


if __name__ == '__main__':
    main()