import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the trained model, selected features, and dataset
@st.cache_resource
def load_model_and_features():
    try:
        model = joblib.load('student_performance_model.pkl')  # Adjust path if needed
        selected_features = joblib.load('selected_features.pkl')  # Load selected features
        encoded_data = pd.read_csv('encoded_feature_engineered_data.csv')  # Load encoded dataset for recommendations
        return model, selected_features, encoded_data
    except Exception as e:
        st.error(f"Error loading model, features, or dataset: {e}")
        return None, None, None

model, selected_features, encoded_data = load_model_and_features()

# Mapping for prediction results
output_mappings = {
    "Early_Dropout_Risk": {0: "Low Risk", 1: "High Risk"},
    "Skill_Gap_Analysis": {0: "No Gap", 1: "Moderate Gap", 2: "Significant Gap"},
    "Study_Group": {0: "Group A", 1: "Group B", 2: "Group C"},
    "Engagement_Prediction": {0: "Low Engagement", 1: "Moderate Engagement", 2: "High Engagement"},
    "Course_Recommendation": {0: "Foundational Courses", 1: "Intermediate Courses", 2: "Advanced Courses"},
    "Study_Schedule": {0: "Light Schedule", 1: "Regular Schedule", 2: "Intensive Schedule"},
    "Recommended_Resources_Self-Study Materials": {0: "Not Recommended", 1: "Recommended"},
    "Recommended_Resources_Study Groups and Peer Learning": {0: "Not Recommended", 1: "Recommended"}
}

# Preprocess user input
def preprocess_input(data):
    try:
        # Feature engineering
        if 'Attendance' in data.columns and 'Hours_Studied' in data.columns:
            data['Is_Regular'] = data['Attendance'].apply(lambda x: 1 if x >= 75 else 0)
            data['Attendance_Study_Ratio'] = data['Attendance'] / (data['Hours_Studied'] + 1e-5)
            data['Attendance_Study_Ratio'] = np.log1p(data['Attendance_Study_Ratio'].abs()) * np.sign(data['Attendance_Study_Ratio'])
            data['Hours_Studied_Squared'] = data['Hours_Studied'] ** 2

        if 'Peer_Influence_Positive' in data.columns and 'Parental_Involvement' in data.columns:
            data['Combined_Score'] = (data['Peer_Influence_Positive'] + data['Parental_Involvement']) / 2

        if 'Sleep_Hours' in data.columns:
            data['Sleep_Quality'] = data['Sleep_Hours'].apply(lambda x: 0 if x < 6 else (0.5 if 6 <= x <= 8 else 1))

        # Retain only selected features
        data = data[selected_features]
        return data
    except Exception as e:
        st.error(f"Error during preprocessing: {e}")
        return None

def generate_recommendations(predicted_values, top_n=5):
    try:
        # Features required for recommendation
        recommendation_features = [
            'Early_Dropout_Risk',
            'Skill_Gap_Analysis',
            'Study_Group',
            'Engagement_Prediction'
        ]

        # Create a DataFrame for the predicted values
        predicted_df = pd.DataFrame([predicted_values], columns=recommendation_features)

        # Select the required recommendation features from the encoded dataset
        recommendation_data = encoded_data[recommendation_features]

        # Compute cosine similarity
        similarity_scores = cosine_similarity(predicted_df, recommendation_data)

        # Find the most similar students
        similar_students = similarity_scores.argsort()[0][::-1][:top_n]

        # Extract recommendations
        recommended_courses = [
            output_mappings["Course_Recommendation"][encoded_data.iloc[i]['Course_Recommendation']]
            for i in similar_students
        ]
        recommended_schedules = [
            output_mappings["Study_Schedule"][encoded_data.iloc[i]['Study_Schedule']]
            for i in similar_students
        ]
        recommended_resources = encoded_data.iloc[similar_students][[
            'Recommended_Resources_Self-Study Materials',
            'Recommended_Resources_Study Groups and Peer Learning'
        ]].idxmax(axis=1).tolist()

        recommendations = {
            "Courses": recommended_courses,
            "Study_Schedules": recommended_schedules,
            "Resources": recommended_resources
        }
        return recommendations
    except Exception as e:
        st.error(f"Error generating recommendations: {e}")
        return None



# Main app logic
def main():
    st.title("Student Performance Prediction with Recommendations")
    st.write("Enter the student's details below to predict their performance and get personalized recommendations.")

    # Create input fields for all possible features
    inputs = {
        
        'Hours_Studied': st.number_input("Hours Studied", min_value=0.0, max_value=24.0, step=0.1),
        'Attendance': st.number_input("Attendance (%)", min_value=0.0, max_value=100.0, step=0.1),
        'Parental_Involvement': st.selectbox("Parental Involvement (1-Low, 3-High)", [1, 2, 3]),
        'Access_to_Resources': st.selectbox("Access to Resources (1-Low, 3-High)", [1, 2, 3]),
        'Extracurricular_Activities': st.selectbox("Extracurricular Activities (0-No, 1-Yes)", [0, 1]),
        'Sleep_Hours': st.number_input("Sleep Hours", min_value=0.0, max_value=24.0, step=0.1),
        'Previous_Scores': st.number_input("Previous Scores (%)", min_value=0.0, max_value=100.0, step=0.1),
        'Motivation_Level': st.selectbox("Motivation Level (1-Low, 3-High)", [1, 2, 3]),
        'Internet_Access': st.selectbox("Internet Access (0-No, 1-Yes)", [0, 1]),
        'Tutoring_Sessions': st.number_input("Tutoring Sessions", min_value=0, max_value=10, step=1),
        'Family_Income': st.selectbox("Family Income (1-Low, 3-High)", [1, 2, 3]),
        'Teacher_Quality': st.selectbox("Teacher Quality (1-Low, 3-High)", [1, 2, 3]),
        'School_Type': st.selectbox("School Type (0-Private, 1-Public)", [0, 1]),
        'Physical_Activity': st.number_input("Physical Activity (Hours per week)", min_value=0.0, max_value=10.0, step=0.1),
        'Learning_Disabilities': st.selectbox("Learning Disabilities (0-No, 1-Yes)", [0, 1]),
        'Exam_Score': st.number_input("Exam Score (%)", min_value=0.0, max_value=100.0, step=0.1),
        'Peer_Influence_Neutral': st.selectbox("Peer Influence Neutral (0-No, 1-Yes)", [0, 1]),
        'Peer_Influence_Positive': st.selectbox("Peer Influence Positive (0-No, 1-Yes)", [0, 1]),
        'Is_Regular': st.selectbox("Is Regular (0-No, 1-Yes)", [0, 1]),
        'Attendance_Study_Ratio': st.number_input("Attendance Study Ratio", min_value=0.0, max_value=100.0, step=0.1),
        'Hours_Studied_Squared': st.number_input("Hours Studied Squared", min_value=0.0, max_value=576.0, step=0.1),
        'Combined_Score': st.number_input("Combined Score", min_value=0.0, max_value=100.0, step=0.1),
        'Sleep_Quality': st.selectbox("Sleep Quality (0=Poor, 0.5=Average, 1=Good)", [0, 0.5, 1]),
        'Performance_Indicator': st.number_input("Performance Indicator", min_value=0.0, max_value=1000.0, step=0.1)
    }


    # Convert inputs into DataFrame
    input_df = pd.DataFrame([inputs])
    
    # Predict button
    if st.button("Predict"):
        if model and selected_features:
            processed_input = preprocess_input(input_df)
            if processed_input is not None:
                try:
                    # Make predictions
                    predictions = model.predict(processed_input)


                    # Map predictions
                    mapped_results = {
                        feature: output_mappings[feature][predictions[0][i]]
                        for i, feature in enumerate(output_mappings)
                    }

                    st.subheader("Prediction Results:")
                    for feature, result in mapped_results.items():
                        st.write(f"{feature}: {result}")

                    # Prepare predicted values for recommendations
                    recommendation_inputs = predictions[0][:4]  # Use first 4 predictions
                    recommendations = generate_recommendations(recommendation_inputs)   

                    if recommendations:
                        st.subheader("Personalized Recommendations:")
                        st.write(f"Recommended Courses: {recommendations['Courses']}")
                        st.write(f"Recommended Study Schedules: {recommendations['Study_Schedules']}")
                        st.write(f"Recommended Resources: {recommendations['Resources']}")

                except Exception as e:
                    st.error(f"Prediction failed: {e}")

        else:
            st.error("Model or selected features not loaded. Check your setup.")

   

if __name__ == "__main__":
    main()
