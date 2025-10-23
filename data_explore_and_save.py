import pandas as pd
from sklearn.datasets import fetch_california_housing

def main():
    data=fetch_california_housing(as_frame=True)
    df=data.frame 
    print("Columns:" , df.columns.tolist())
    print("Shape", df.shape)
    print(df.head())

    print("\n Summary statistics:")
    print(df.describe().T)

    df.to_csv("california_housing.csv", index=False)
    print("\n Saved dataset to california_housing.csv")

if __name__ == "__main__":
    main()



