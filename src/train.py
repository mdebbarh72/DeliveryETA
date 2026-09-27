import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
import joblib

def main():
    df = pd.read_csv('data/bronze/FoodDeliveryClean.csv')
    df = df.drop(columns=['Order_ID'])

    X = df.drop(columns=['Delivery_Time_min'])
    y = df['Delivery_Time_min']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cat_features = ['Weather', 'Traffic_Level', 'Time_of_Day', 'Vehicle_Type']
    num_features = ['Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ])

    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', XGBRegressor(random_state=42))
    ])

    param_distributions = {
        'model__n_estimators': [100, 200, 300, 500],
        'model__learning_rate': [0.01, 0.05, 0.1, 0.2],
        'model__max_depth': [3, 5, 7, 9],
        'model__subsample': [0.6, 0.8, 1.0],
        'model__colsample_bytree': [0.6, 0.8, 1.0]
    }

    random_search = RandomizedSearchCV(pipeline, param_distributions, n_iter=30, 
                                       cv=5, scoring='r2', n_jobs=-1, random_state=42)
    
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_

    def evaluate(model, X, y, dataset_name):
        y_pred = model.predict(X)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        n = X.shape[0]
        p = X.shape[1]
        adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)
        print(f"--- {dataset_name} ---")
        print(f"RMSE: {rmse:.2f} min")
        print(f"MAE: {mae:.2f} min")
        print(f"R²: {r2:.4f}")
        print(f"Adjusted R²: {adj_r2:.4f}")
        return mae, r2

    evaluate(best_model, X_train, y_train, "Train Set")
    mae_test, r2_test = evaluate(best_model, X_test, y_test, "Test Set")

    print("\n--- Interprétation Métier ---")
    print(f"Le modèle se trompe en moyenne de {mae_test:.2f} minutes sur ses prédictions de temps de livraison.")

    joblib.dump(best_model, 'models/delivery_model.pkl')
    print("Modèle sauvegardé dans models/delivery_model.pkl")

if __name__ == "__main__":
    main()
