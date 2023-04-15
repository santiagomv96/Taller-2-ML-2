import numpy as np

class K_Means:

    def __init__(self,X=None,K=None,max_iters=1000):
        self.X=X
        self.K=K
        self.max_iters=max_iters
    
    def fit(self,X,K,max_iters):
        self.X=X
        self.K=K
        self.max_iters=max_iters

        #gt burn first centroids      
        self.centroids = self.X[np.random.choice(len(self.X), self.K, replace=False)]
        for i in range(self.max_iters):
            # Asignación de clúster
            # miro la distancia euclidiana entre los registros (X,col) y centroides (K,col) y llevo todo
            # a una matriz de 3 dimensiones K,x,col y suma dichas distancias, generando una nueva matriz
            # distancias (k,X) que indica la distancia del punto al centroide
            distances = np.sqrt(((self.X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            # Recomputación de centroides
            for j in range(self.K):
                self.centroids[j] = np.mean(self.X[self.labels == j], axis=0)

    def transform(self,X):
        distances = []
        for c in self.centroids:
            distances.append(np.linalg.norm(X - c, axis=1))  # Calcula la distancia entre cada punto de datos y el centroide c

        distances = np.array(distances)  # Convierte a una matriz de forma (k, n)
        return  np.argmin(distances, axis=0)  # Etiqueta cada punto de datos con el clúster correspondiente al centroide más cercano
        
    def fit_transform(self,X,K,max_iters):    
        self.fit(X,K,max_iters)
        return self.transform(X)
