import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def transformation(train_df, test_df):
    train_df["Traffic_Level_Encoded"] = train_df["Road_traffic_density"].map(traffic_order)
    test_df["Traffic_Level_Encoded"] = test_df["Road_traffic_density"].map(traffic_order)
    train_df.drop(columns=["Road_traffic_density"], inplace=True)
    test_df.drop(columns=["Road_traffic_density"], inplace=True)

    categorical_cols = ["Type_of_vehicle", "Time_of_Day"]

    ohe = OneHotEncoder(sparse_output=False, drop=None, handle_unknown="ignore")
    ohe.fit(train_df[categorical_cols])

    train_ohe = pd.DataFrame(
        ohe.transform(train_df[categorical_cols]),
        columns=ohe.get_feature_names_out(categorical_cols),
        index=train_df.index,
    )
    test_ohe = pd.DataFrame(
        ohe.transform(test_df[categorical_cols]),
        columns=ohe.get_feature_names_out(categorical_cols),
        index=test_df.index,
    )

    train_df = pd.concat([train_df, train_ohe], axis=1).drop(columns=categorical_cols)
    test_df = pd.concat([test_df, test_ohe], axis=1).drop(columns=categorical_cols)

    numeric_cols = [
        "Delivery_Person_Age",
        "Delivery_Person_Rating",
        "distance",
        "Time_Taken",
        "Delivery_Person_Experience",
    ]

    scaler = StandardScaler()
    scaler.fit(train_df[numeric_cols])

    train_df[numeric_cols] = scaler.transform(train_df[numeric_cols])
    test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])

    print(train_df.head())
    print(test_df.head())

    train_df.to_csv("../data/delivery_data_train_processed.csv", index=False)
    test_df.to_csv("../data/delivery_data_test_processed.csv", index=False)
    print("\nSaved processed train/test data")
