import pandas as pd
import statsmodels.api as sm # import statsmodels 
df=pd.read_csv("https://s3.amazonaws.com/hr-testcases/399/assets/trainingdata.txt",sep=',',header=None)
df.columns=['X','y']



X = df["X"] ## X usually means our input variables (or independent variables)
y = df["y"] ## Y usually means our output/dependent variable
X = sm.add_constant(X) ## let's add an intercept (beta_0) to our model

# Note the difference in argument order
model = sm.OLS(y, X).fit() ## sm.OLS(output, input)
predictions = model.predict(X)

# Print out the statistics
model.summary()
