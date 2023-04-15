import numpy as np

class T_SNE:

       

    def __init__(self,x,n_components, perplexity=30.0, learning_rate=200.0, n_iter=1000, tol=1e-5):   
        self.x=x 
        self.perplexity=perplexity
        self.tol=tol
        self.n_components=n_components
        self.learning_rate=learning_rate
        self.n_iter=n_iter
        self.n = self.x.shape[0]
        self.P = self.calculate_P()
        Y = np.random.randn(self.n, self.n_components)
        for i in range(n_iter):
            cost = self.compute_cost(self.P, Q)
            grad = self.compute_gradient(self.P, Y)
            Y -= learning_rate * grad
            Q = self.calculate_Q(Y)
        return Y



        def euclidean_distance(self,v1, v2):
            '''Computes de distance between two vectors'''
            return np.sqrt(np.sum((v1 - v2) ** 2))
    
        def calculate_pairwise_distances(self):
            '''Compute a distances matrix, takes all pair of possible pouinst mixes and check its 
            distance'''
            self.distances = np.zeros((self.n, self.n))
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    self.distances[i, j] = euclidean_distance(self.x[i], self.x[j])
                    self.distances[j, i] = self.distances[i, j]

        def calculate_P(self):
            '''creates a probability matrix(n*n), takes every point and computes de probability that
            another point be its neighboor on a gaussian distribution'''
            pairwise_distances = calculate_pairwise_distances(self.x)
            self.P = np.zeros((self.n, self.n))
            for i in range(self.n):
                #Beta is a parameter to find a correct sigma with a given perplexy, 
                # beta will be in betamin and betamax, betamin and betamax stars whit big 
                # values, this will be modifiers
                beta = 1.0
                betamin = -np.inf
                betamax = np.inf
                #Compute Distance matrix Di, taking form i all the indexes from the other points and
                #concatenate this on Di
                Di = pairwise_distances[i, np.concatenate((np.r_[0:i], np.r_[i+1:self.n]))]
                #Shanon entropy and thisP is de afinity matrix, whit H we look for an specific perplexity
                H, thisP = entropy(Di, beta)
                Hdiff = H - perplexity
                tries = 0
                #if the Shanon entropy is too big it come to modify beta
                while np.abs(Hdiff) > tol and tries < 50:
                    if Hdiff > 0:
                        #the beta limits are delimited
                        betamin = beta
                        #ask if betamax is infinite
                        if np.isinf(betamax):
                            beta = beta * 2.
                        else:
                            beta = (beta + betamax) / 2.
                    else:
                        #the beta limits are delimited
                        betamax = beta
                        #ask if betamin is infinite
                        if np.isinf(betamin):
                            beta = beta / 2.
                        else:
                            beta = (beta + betamin) / 2.
                    #recalculates shanon entrpy and afinity matrix
                    H, thisP = entropy(Di, beta)                    
                    Hdiff = H - perplexity
                    tries += 1
                self.P[i, np.concatenate((np.r_[0:i], np.r_[i+1:self.n]))] = thisP

        def entropy(self,D, beta):
            '''Shanon entropy'''
            prob = np.exp(-D * beta)
            sumprob = np.sum(prob)
            H = np.log(sumprob) + beta * np.sum(D * prob) / sumprob
            prob = prob / sumprob
            return H, prob
        
        def compute_gradient(self, Y):
            '''the error gradient, it calculates the difference between the afinity matrix with all 
            the features and the the afinity matrix withot all the features, it modify de the points 
            location'''
            d = Y.shape[1]
            self.Q = np.zeros((self.n, self.n))
            PQ = np.zeros((self.n, self.n))
            for i in range(self.n):
                for j in range(i + 1, self.n):
                    self.Q[i, j] = 1.0 / (1.0 + np.sum((Y[i] - Y[j]) ** 2))
                    self.Q[j, i] = self.Q[i, j]
                    PQ[i, j] = self.P[i, j] - self.Q[i, j]
                    PQ[j, i] = self.P[j, i] - self.Q[j, i]
            grad = np.zeros((self.n, d))
            for i in range(self.n):
                for j in range(self.n):
                    grad[i] += 4 * (PQ[i, j] * (Y[i] - Y[j]) * self.Q[i, j])
            return grad

        def compute_cost(self):
            ''' compute the cost function, is the distance between de matrix with all features
            and reduced matrix'''
            cost = np.sum(self.P * np.log(self.P / self.Q))
            return cost    