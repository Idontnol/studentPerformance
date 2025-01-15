# import streamlit as st
# import joblib
# import numpy as np

# # Define a function to load the trained model
# @st.cache_resource
# def load_model():
#     model = joblib.load('student_performance_model.pkl')
#     return model

# # Load the model
# try:
#     model = load_model()
# except Exception as e:
#     st.error(f"Error loading the model: {e}")
#     model = None

# # Define the prediction function
# def predict_performance(attendance, hours_studied, motivation_level, previous_scores):
#     input_data = np.array([[attendance, hours_studied, motivation_level, previous_scores]])
#     try:
#         prediction = model.predict(input_data)
#         return prediction
#     except Exception as e:
#         st.error(f"Error making prediction: {e}")
#         return None

# # Create the Streamlit app
# def main():
#     st.title("Student Performance Prediction")
    
#     st.write("Enter the student's details below to predict their performance.")

#     # Create input fields
#     attendance = st.number_input("Attendance", min_value=0.0, max_value=100.0, step=0.1)
#     hours_studied = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, step=0.1)
#     motivation_level = st.number_input("Motivation Level", min_value=0.0, max_value=10.0, step=0.1)
#     previous_scores = st.number_input("Previous Scores", min_value=0.0, max_value=100.0, step=0.1)

#     if st.button("Predict"):
#         if model is not None:
#             # Make prediction
#             prediction = predict_performance(attendance, hours_studied, motivation_level, previous_scores)
            
#             if prediction is not None:
#                 # Display the prediction
#                 st.subheader("Prediction Results:")
#                 st.write(f"Early Dropout Risk: {prediction[0][0]}")
#                 st.write(f"Skill Gap Analysis: {prediction[0][1]}")
#                 st.write(f"Study Group: {prediction[0][2]}")
#                 st.write(f"Engagement Prediction: {prediction[0][3]}")
#                 st.write(f"Course Recommendation: {prediction[0][4]}")
#                 st.write(f"Study Schedule: {prediction[0][5]}")
#                 st.write(f"Recommended Resources (Self-Study Materials): {prediction[0][6]}")
#                 st.write(f"Recommended Resources (Study Groups and Peer Learning): {prediction[0][7]}")
#         else:
#             st.error("Model is not loaded. Please check the logs for more details.")

# if __name__ == '__main__':
#     main()
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Define a function to load the trained model
@st.cache_resource
def load_model():
    model = joblib.load('student_performance_model.pkl')
    return model

# Load the model
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading the model: {e}")
    model = None

# Define the preprocessing function
def preprocess_input(data):
    # Apply StandardScaler to numerical columns
    scaler = StandardScaler()
    numerical_columns = data.select_dtypes(include=['int64', 'float64']).columns
    data[numerical_columns] = scaler.fit_transform(data[numerical_columns])
    
    # Create engineered features
    if 'Attendance' in data.columns and 'Hours_Studied' in data.columns:
        data['Is_Regular'] = data['Attendance'].apply(lambda x: 1 if x >= 75 else 0)
        data['Attendance_Study_Ratio'] = data['Attendance'] / (data['Hours_Studied'] + 1e-5)
        data['Attendance_Study_Ratio'] = np.log1p(data['Attendance_Study_Ratio'].abs()) * np.sign(data['Attendance_Study_Ratio'])
        data['Hours_Studied_Squared'] = data['Hours_Studied'] ** 2

    if 'Peer_Influence_Positive' in data.columns and 'Parental_Involvement' in data.columns:
        data['Combined_Score'] = (data['Peer_Influence_Positive'] + data['Parental_Involvement']) / 2

    if 'Sleep_Hours' in data.columns:
        data['Sleep_Quality'] = data['Sleep_Hours'].apply(lambda x: 0 if x < 6 else (0.5 if 6 <= x <= 8 else 1))
    
    return data

# Define the prediction function
def predict_performance(features):
    try:
        processed_features = preprocess_input(features)
        prediction = model.predict(processed_features)
        return prediction
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

# Create the Streamlit app
def main():
    st.title("Student Performance Prediction")
    
    st.write("Enter the student's details below to predict their performance.")

    # Create input fields for all the features in the specified order
    hours_studied = st.number_input("Hours Studied", min_value=0.0, max_value=24.0, step=0.1, key="hours_studied")
    attendance = st.number_input("Attendance", min_value=0.0, max_value=100.0, step=0.1, key="attendance")
    parental_involvement = st.number_input("Parental Involvement", min_value=1, max_value=3, step=1, key="parental_involvement")
    access_to_resources = st.number_input("Access to Resources", min_value=1, max_value=3, step=1, key="access_to_resources")
    extracurricular_activities = st.number_input("Extracurricular Activities", min_value=0, max_value=1, step=1, key="extracurricular_activities")
    sleep_hours = st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, step=0.1, key="sleep_hours")
    previous_scores = st.number_input("Previous Scores", min_value=0.0, max_value=100.0, step=0.1, key="previous_scores")
    motivation_level = st.number_input("Motivation Level", min_value=0.0, max_value=10.0, step=0.1, key="motivation_level")
    internet_access = st.number_input("Internet Access", min_value=0, max_value=1, step=1, key="internet_access")
    tutoring_sessions = st.number_input("Tutoring Sessions", min_value=0, max_value=10, step=1, key="tutoring_sessions")
    family_income = st.number_input("Family Income", min_value=1, max_value=3, step=1, key="family_income")
    teacher_quality = st.number_input("Teacher Quality", min_value=1, max_value=3, step=1, key="teacher_quality")
    school_type = st.number_input("School Type", min_value=0, max_value=1, step=1, key="school_type")
    physical_activity = st.number_input("Physical Activity", min_value=0.0, max_value=10.0, step=0.1, key="physical_activity")
    learning_disabilities = st.number_input("Learning Disabilities", min_value=0, max_value=1, step=1, key="learning_disabilities")
    exam_score = st.number_input("Exam Score", min_value=0.0, max_value=100.0, step=0.1, key="exam_score")
    peer_influence_neutral = st.number_input("Peer Influence Neutral", min_value=0, max_value=1, step=1, key="peer_influence_neutral")
    peer_influence_positive = st.number_input("Peer Influence Positive", min_value=0, max_value=1, step=1, key="peer_influence_positive")
    is_regular = st.number_input("Is Regular", min_value=0, max_value=1, step=1, key="is_regular")
    attendance_study_ratio = st.number_input("Attendance Study Ratio", min_value=0.0, max_value=100.0, step=0.1, key="attendance_study_ratio")
    hours_studied_squared = st.number_input("Hours Studied Squared", min_value=0.0, max_value=100.0, step=0.1, key="hours_studied_squared")
    combined_score = st.number_input("Combined Score", min_value=0.0, max_value=100.0, step=0.1, key="combined_score")
    sleep_quality = st.number_input("Sleep Quality", min_value=0, max_value=1, step=1, key="sleep_quality")
    performance_indicator = st.number_input("Performance Indicator", min_value=0.0, max_value=100.0, step=0.1, key="performance_indicator")

    if st.button("Predict"):
        if model is not None:
            # Gather all the input features into a DataFrame
            input_data = pd.DataFrame({
                'Hours_Studied': [hours_studied],
                'Attendance': [attendance],
                'Parental_Involvement': [parental_involvement],
                'Access_to_Resources': [access_to_resources],
                'Extracurricular_Activities': [extracurricular_activities],
                'Sleep_Hours': [sleep_hours],
                'Previous_Scores': [previous_scores],
                'Motivation_Level': [motivation_level],
                'Internet_Access': [internet_access],
                'Tutoring_Sessions': [tutoring_sessions],
                'Family_Income': [family_income],
                'Teacher_Quality': [teacher_quality],
                'School_Type': [school_type],
                'Physical_Activity': [physical_activity],
                'Learning_Disabilities': [learning_disabilities],
                'Exam_Score': [exam_score],
                'Peer_Influence_Neutral': [peer_influence_neutral],
                'Peer_Influence_Positive': [peer_influence_positive],
                'Is_Regular': [is_regular],
                'Attendance_Study_Ratio': [attendance_study_ratio],
                'Hours_Studied_Squared': [hours_studied_squared],
                'Combined_Score': [combined_score],
                'Sleep_Quality': [sleep_quality],
                'Performance_Indicator': [performance_indicator]
            })

            # Make prediction
            prediction = predict_performance(input_data)
            
            if prediction is not None:
                # Display the prediction
                st.subheader("Prediction Results:")
                st.write(f"Early Dropout Risk: {prediction[0][0]}")
                st.write(f"Skill Gap Analysis: {prediction[0][1]}")
                st.write(f"Study Group: {prediction[0][2]}")
                st.write(f"Engagement Prediction: {prediction[0][3]}")
                st.write(f"Course Recommendation: {prediction[0][4]}")
                st.write(f"Study Schedule: {prediction[0][5]}")
                st.write(f"Recommended Resources (Self-Study Materials): {prediction[0][6]}")
                st.write(f"Recommended Resources (Study Groups and Peer Learning): {prediction[0][7]}")
        else:
            st.error("Model is not loaded. Please check the logs for more details.")

if __name__ == '__main__':
    main()