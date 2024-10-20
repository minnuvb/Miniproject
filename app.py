import streamlit as st
import pandas as pd
import pickle
import base64

# Function to load the selected model
@st.cache_resource
def load_linear_regression_model():
    try:
        with open('linear_regression_model.pkl', 'rb') as file:
            linear_model = pickle.load(file)
        return linear_model
    except FileNotFoundError:
        st.error("The Linear Regression model file was not found.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the Linear Regression model: {e}")
        return None

@st.cache_resource
def load_decision_tree_model():
    try:
        with open('decision_tree_regressor_model.pkl', 'rb') as file:
            decision_tree_model = pickle.load(file)
        return decision_tree_model
    except FileNotFoundError:
        st.error("The Decision Tree model file was not found.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the Decision Tree model: {e}")
        return None

@st.cache_resource
def load_random_forest_model():
    try:
        with open('random_forest_regressor_model.pkl', 'rb') as file:
            random_forest_model = pickle.load(file)
        return random_forest_model
    except FileNotFoundError:
        st.error("The Random Forest model file was not found.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the Random Forest model: {e}")
        return None

@st.cache_resource
def load_xgb_model():
    try:
        with open('xgb_model.pkl', 'rb') as file:
            xgb_model = pickle.load(file)
        return xgb_model
    except FileNotFoundError:
        st.error("The XGBoost model file was not found.")
        return None
    except Exception as e:
        st.error(f"An error occurred while loading the XGBoost model: {e}")
        return None



# **Sidebar for navigation**
st.sidebar.title('**Navigation**')
options = st.sidebar.selectbox('**Select a page:**', 
                               ['Prediction', 'Code', 'About'])

# Model selection and prediction section
if options == 'Prediction':
    st.title("**Calories Burnt Prediction Web App**")
    
    # **Model selection dropdown**
    model_choice = st.selectbox('**Select Model**', 
                                ['Linear Regression', 'Decision Tree Regressor', 'Random Forest Regressor', 'XGBoost Regressor'])
    
    # Dictionary mapping for model loading functions
    model_loaders = {
        'Linear Regression': load_linear_regression_model,
        'Decision Tree Regressor': load_decision_tree_model,
        'Random Forest Regressor': load_random_forest_model,
        'XGBoost Regressor': load_xgb_model
    }

    # Load the selected model
    model = model_loaders[model_choice]()

    # Ensure the model is loaded before proceeding
    if model:
        # **User input fields**
        gender = st.selectbox('**Gender**', ['Male', 'Female'])
        age = st.number_input('**Age**', min_value=1, max_value=100, value=25)
        height = st.number_input('**Height (cm)**', min_value=50, max_value=250, value=170)
        weight = st.number_input('**Weight (kg)**', min_value=1, max_value=200, value=70)
        duration = st.number_input('**Duration of Activity (minutes)**', min_value=1, max_value=300, value=30)
        heart_rate = st.number_input('**Heart Rate (bpm)**', min_value=40, max_value=200, value=120)
        body_temp = st.number_input('**Body Temperature (°C)**', min_value=30, max_value=45, value=37)

        # Convert gender to numerical value
        gender_num = 1 if gender == 'Male' else 0

        # Prepare input data
        input_data = pd.DataFrame([[gender_num, age, height, weight, duration, heart_rate, body_temp]], 
                                  columns=["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"])

        # **Prediction button only in Prediction section**
        if st.button('**Predict**'):
            prediction = model.predict(input_data)  # Use the loaded model for prediction
            st.markdown(f'**The predicted Calories Burnt is: {prediction[0]:,.2f}**')  # Display prediction with bold

            with st.expander("**Show more details**"):
                st.write("**Details of the prediction:**")
                st.write(f'Model used: {model_choice}')

# **Code section**
elif options == 'Code':
    st.header('**Code**')
    # **Add a button to download the Jupyter notebook (.ipynb) file**
    notebook_path = 'calorie_burnt_prediction.ipynb'
    with open(notebook_path, "rb") as file:
        btn = st.download_button(
            label="Download Jupyter Notebook",
            data=file,
            file_name="calories_burnt_prediction.ipynb",
            mime="application/x-ipynb+json"
        )
    st.write('You can download the Jupyter notebook to view the code and the model building process.')
    st.write('--'*50)

    st.header('**Data**')
    # **Add a button to download your dataset**
    data_path = 'calories.csv'
    with open(data_path, "rb") as file:
        btn = st.download_button(
            label="Download Dataset",
            data=file,
            file_name="calories.csv",
            mime="text/csv"
        )
    st.write('You can download the dataset to use it for your own analysis or model building.')
    st.write('--'*50)

    st.header('**GitHub Repository**')
    st.write('You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/minnuvb/Miniproject)')
    st.write('--'*50)

# **About section**
elif options == 'About':
    st.title('**About**')
    st.write('This web app is created to predict the calories burnt based on the user inputs such as gender, age, height, weight, duration, heart rate, and body temperature.')
    st.write('The model used in this web app is a XGBoost model trained on the dataset with 15000 samples.')
    st.write('The dataset used in this web app is collected from the Kaggle dataset: [Dataset](https://www.kaggle.com/datasets/fmendes/fmendesdat263xdemos).')

    st.write('The web app is open-source. You can view the code and the dataset used in this web app from the GitHub repository:')
    st.write('[GitHub Repository](https://github.com/minnuvb/Miniproject)')
    st.write('--'*50)

    st.header('**Contact**')
    st.write('You can contact me for any queries or feedback:')
    st.write('**Email:** minnuvb97@gmail.com')
    st.write('**LinkedIn:** [Minnu VB](https://www.linkedin.com/in/minnu-v-b-868040130/)')
    st.write('--'*50)
