import numpy as np
import pandas as pd
from sklearn import linear_model as lm
from sklearn import preprocessing as p

F, N = map(int, input().split())
train = np.array([input().split() for _ in range(N)], float)
T = int(input())
test = np.array([input().split() for _ in range(T)], float)

mod = lm.LinearRegression()
X = p.PolynomialFeatures(3, include_bias=False)
mod.fit(X.fit_transform(train[:, :-1]), train[:, -1])

ymod = mod.predict(X.fit_transform(test))
print(*ymod, sep='\n')