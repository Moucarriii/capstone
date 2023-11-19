import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
import pickle

def train_and_save_model(data_path, model_path):
    # Load the dataset
    df = pd.read_csv(data_path)

    # Separate features and target variable
    X = df.drop('fee_percentage', axis=1)
    y = df['fee_percentage']

    # Split the data into training and test sets
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    # Define categorical and numeric columns
    categorical_cols = X_train.select_dtypes(include=['object', 'category', 'bool']).columns
    numeric_cols = X_train.select_dtypes(include=['int64', 'float64']).columns

    # Create the preprocessing pipelines for both numeric and categorical data
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ])

    # Instantiate the Random Forest Regressor
    model = RandomForestRegressor(
        n_estimators=400, 
        max_depth=None, 
        min_samples_split=2, 
        min_samples_leaf=1, 
        max_features='sqrt', 
        random_state=42)

    # Create a pipeline with the preprocessor and the model
    final_pipeline = Pipeline(steps=[('preprocessor', preprocessor), 
                                     ('model', model)])

    # Fit the model
    final_pipeline.fit(X_train, y_train)

    # Save the trained model
    with open(model_path, 'wb') as file:
        pickle.dump(final_pipeline, file)

if __name__ == '__main__':
    data_path = "Data-11Col.csv"
    model_path = "model.pkl"
    train_and_save_model(data_path, model_path)
