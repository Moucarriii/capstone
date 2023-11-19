import streamlit as st
import pandas as pd
import pickle

# Function to load the model
def load_model(model_path):
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
    return model

def main():
    # Set page title and favicon
    st.set_page_config(page_title="ML Model Prediction", page_icon=":bar_chart:", layout="wide")

    # Load the model
    model_path = "C:\\Users\\ADMIN\\Desktop\\AIKIDO\\New-Streamlit\\model.pkl"
    model = load_model(model_path)

    # Title and introduction
    st.title("Machine Learning Model Prediction")
    st.markdown("""
        Use the input fields below to enter the details for the prediction. 
        Once you've filled out all the fields, press the 'Predict' button.
    """)

    # Form for user input
    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            amount_financed = st.number_input('Amount Financed', min_value=0.0, format="%.2f")
            max_client_financed_amount = st.number_input('Max Client Financed Amount', min_value=0.0, format="%.2f")
            client_transactions = st.number_input('Client Transactions', min_value=0)

        with col2:
            days_till_payment = st.number_input('Days Till Payment', min_value=0)
            client_size = st.selectbox('Client Size', ['Medium', 'Small', 'No Information', 'Large', 'Micro Business'])
            debtor_transactions = st.number_input('Debtor Transactions', min_value=0)

        with col3:
            average_client_spread = st.number_input('Average Client Spread', min_value=0.0, format="%.2f")
            debtor_size = st.selectbox('Debtor Size', ['Medium', 'Small', 'No Information', 'Large', 'Micro Business'])
            debtor_tenure = st.number_input('Debtor Tenure', min_value=0)

        debtor_type = st.selectbox('Debtor Type', ["private", "public"])
        client_recency = st.number_input('Client Recency', min_value=0)

        submit_button = st.form_submit_button("Predict")

    if submit_button:
        data = pd.DataFrame({
            'amount_financed': [amount_financed],
            'days_till_payment': [days_till_payment],
            'max_client_financed_amount': [max_client_financed_amount],
            'client_size': [client_size],
            'debtor_size': [debtor_size],
            'debtor_type': [debtor_type],
            'client_transactions': [client_transactions],
            'debtor_transactions': [debtor_transactions],
            'average_client_spread': [average_client_spread],
            'debtor_tenure': [debtor_tenure],
            'client_recency': [client_recency]
        })

        prediction = model.predict(data)
        st.success(f'Predicted Fee Percentage: {prediction[0]:.2f}%')

if __name__ == '__main__':
    main()
