import numpy as np

class LRegression(object):
    def __init__(self, num_dim, num_label, regular_v = 0.01):
        # The dimension of a sample case  
        self.num_dim = num_dim
        # The number of label
        self.num_label = num_label
        # A penalty term for L2-norm regularization
        self.regular_v = regular_v

        # Initilize a weight matrix randomly, W, num_dim X num_label
        # with a Gaussion distribution with mean 0 and standard
        # deviation
        self.weights = 0.1 * np.random.randn(self.num_dim, self.num_label)
        # Insert a weight row for bias unit.
        self.weights = np.insert(self.weights, 0, 0, axis = 0)

    def train(self, data_X, data_Y, max_epochs = 1000):
        """
        Train a logistic regression model with stochastic gradient descent

        Parameters:
        ----------
        data, a matrix where each row is a training example
        """
        num_cases = data_X.shape[0]

        # Insert bias units of 1 into the first colum
        data_X = np.insert(data_X, 0, 1, axis = 1)

        for epoch in range(max_epochs):
            [cost, grad] = self.getCostGrad(data_X, data_Y, epoch % num_cases)
            # A interative step, decreasing as the iteration time increase.
            alpha = 0.1 / (1.0 + np.sqrt(epoch))
            self.weights += alpha * grad
            #print "%d Iteration cost - %f" % (epoch, cost)
    
    def predict(self, x, threshhold = 0.5):
        x = np.insert(x, 0, 1, axis = 1)
        probs = self._logistic(x)
        return probs > threshhold

    """
        Stochastic gradient descent to update parameters.
    """
    def getCostGrad(self, X, y, i):

        # compute label prob
        lprobs = self._logistic(X[i])
        print lprobs
        
        # The cross-entroy cost function:
        # l(w) = -sum(y[i] * log(probs) + (1 - y[i]) * log(1 - probs)) \
        # + 0.5 * lambda * sum(weights[2]^2)
        cost = np.dot(-y[i], np.log(lprobs).T) + np.dot((y[i] - 1), np.log(1 - lprobs).T) \
                + 0.5 * self.regular_v * sum(self.weights[1:].flatten()**2)
        # The negative grad function
        # X[i].T * (probs - y[i])
        grad = np.dot(X[i].T, (lprobs - y[i]))

        # add the regularization term except for bias units 
        grad[1:] += self.regular_v * self.weights[1:]

        return cost, grad

    def _logistic(self, x):
        return (1.0 / (1.0 + np.exp(np.dot(x, self.weights))))


if __name__ == "__main__":
    lgression = LRegression(6, 2)
    training_data = np.matrix([[1,1,1,0,0,0],\
                            [1,0,1,0,0,0],\
                            [1,1,1,0,0,0],\
                            [0,0,1,1,1,0],\
                            [0,0,1,1,0,0],\
                            [0,0,1,1,1,0]])
    training_Y = np.matrix([[1, 0],
                            [1, 0],
                            [1, 0],
                            [0, 1],
                            [0, 1],
                            [1, 0]])
    lgression.train(training_data, training_Y)
    for i in range(len(training_data)):
        print lgression.predict(training_data[i])
