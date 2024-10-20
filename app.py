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

# Load background image
image_path = "images.png"
with open(image_path, "rb") as f:
    img = f.read()
background_image = base64.b64encode(img).decode()

# Inject CSS for background image directly into the app
st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/jpeg;base64,{background_image});
        background-size: cover;
        background-repeat: no-repeat;
        background-position: center;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar for navigation
st.sidebar.title('Navigation')
options = st.sidebar.selectbox('Select a page:', 
                               ['Prediction', 'Code', 'About'])

# Model selection
if options == 'Prediction':
    st.title("Calories Burnt Prediction Web App")
    
    # Model selection dropdown
    model_choice = st.selectbox('Select Model', ['Linear Regression', 'Decision Tree Regressor', 'Random Forest Regressor', 'XGBoost Regressor'])
    
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
        # User input fields
        gender = st.selectbox('Gender', ['Male', 'Female'])
        age = st.number_input('Age', min_value=1, max_value=100, value=25)
        height = st.number_input('Height (cm)', min_value=50, max_value=250, value=170)
        weight = st.number_input('Weight (kg)', min_value=1, max_value=200, value=70)
        duration = st.number_input('Duration of Activity (minutes)', min_value=1, max_value=300, value=30)
        heart_rate = st.number_input('Heart Rate (bpm)', min_value=40, max_value=200, value=120)
        body_temp = st.number_input('Body Temperature (°C)', min_value=30, max_value=45, value=37)

        # Convert gender to numerical value
        gender_num = 1 if gender == 'Male' else 0

        # Prepare input data
        input_data = pd.DataFrame([[gender_num, age, height, weight, duration, heart_rate, body_temp]], 
                                  columns=["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"])

        # Prediction button
        if st.button('Predict'):
            prediction = model.predict(input_data)
            st.markdown(f'**The predicted Calories Burnt is: {prediction[0]:,.2f}**')
            
            with st.expander("Show more details"):
                st.write("Details of the prediction:")
                st.write(f'Model used: {model_choice}')
