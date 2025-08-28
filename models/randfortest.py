import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
df = pd.read_csv("ESI-3.csv",sep=',')
df=df.dropna()

X = df.iloc[:, 2:-1].values
y = df.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

X_train=X
y_train=y

rf_model = RandomForestRegressor(n_estimators=100, random_state=20)
rf_model.fit(X_train, y_train)


error=0
for i in range(len(X_test)):
    pre=rf_model.predict(X_test[i].reshape(1, -1))
    error1=abs(pre-y_test[i])/y_test[i]

    error=error+error1
print("Acccuracy :",(1-(error/len(X_test)))*100)