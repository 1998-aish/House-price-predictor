import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def main():
    csv_path = "california_housing.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found. Run data_explore_and_save.py first.")

    df=pd.read_csv(csv_path)
    target="MedHouseVal"

    X=df.drop(columns=[target])
    y=df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Linear Regression baseline
    lr=LinearRegression()
    lr.fit(X_train,y_train)
    lr_pred=lr.predict(X_test)
    print("Linear Regeression RMSE:", rmse(y_test,lr_pred))
    print("Linear Regeression R2:", r2_score(y_test,lr_pred))

    # Random Forest
    rf=RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train,y_train)
    rf_pred=rf.predict(X_test)
    print("\nRandom Forest RMSE: ", rmse(y_test,rf_pred))
    print("\nRandom Forest R2: ", r2_score(y_test,rf_pred))
    
    #choose best
    chosen = rf if r2_score(y_test,rf_pred) > r2_score(y_test,lr_pred) else lr
    joblib.dump(chosen, "house_price_model.joblib")
    print("\nSaved model to house_price_model.joblib")

    # Save evaluation CSV for the app: use predictions from chosen model
    chosen_pred = chosen.predict(X_test)
    eval_df = pd.DataFrame({
        "y_true": y_test.values,
        "y_pred": chosen_pred
    })
    eval_df.to_csv("eval.csv", index=False)
    print("Saved evaluation file eval.csv (first 5 rows):")
    print(eval_df.head())

if __name__ == "__main__":
    main()

