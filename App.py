import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import StandardScaler

@st.cache_data
def load_full_dataset_for_rec():
    df = pd.read_csv("student_habits_performance.csv")
    df.drop(columns=['student_id'], inplace=True)

    label_encoders = joblib.load("label_encoders.pkl")
    for col, le in label_encoders.items():
        if col in df.columns:
            # Handling value NaN
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])
            df[col] = le.transform(df[col])

    df.dropna(inplace=True)

    X_original = df.drop(columns=['exam_score'])
    
    # Scaling full
    full_scaler = StandardScaler().fit(X_original)
    X_original_scaled = pd.DataFrame(full_scaler.transform(X_original), columns=X_original.columns)
    
    return df, X_original_scaled

# Load file
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")
df_original, X_original_scaled_full = load_full_dataset_for_rec()

st.title("🎓 Student Score Predictor & Peer Recommendations")

# User input
st.header("Enter Your Habits & Info")
study_hours = st.slider("📘 Study hours per day", 0.0, 12.0, 2.0, step=0.5)
mental_health_rating = st.slider("🧠 Mental Health Rating (Scale of 1-5)", 1, 5, 4)
social_media_hours = st.slider("📱 Social media usage (hours)", 0.0, 10.0, 3.0, step=0.5)
sleep_hours = st.slider("😴 Sleep hours", 0.0, 12.0, 7.0, step=0.5)
netflix_hours = st.slider("🎬 Movie hours", 0.0, 8.0, 1.0, step=0.5)
exercise_freq = st.slider("🏃‍♂️ Exercise frequency (days/week)", 0, 7, 2, )
attendance = st.slider("🏫 Class attendance (%)", 0, 100, 85)

if st.button("🔍 Predict Score & Get Peer-Based Tips"):
    
    model_features = model.feature_names_in_
    
    user_inputs_for_model = {
        'study_hours_per_day': study_hours,
        'mental_health_rating': mental_health_rating,
        'social_media_hours': social_media_hours,
        'sleep_hours': sleep_hours,
        'netflix_hours': netflix_hours,
        'exercise_frequency': exercise_freq,
        'attendance_percentage': attendance,
    }

    input_df_model = pd.DataFrame([user_inputs_for_model], columns=model_features)
    input_scaled_model = scaler.transform(input_df_model)
    predicted_score = model.predict(input_scaled_model)[0]

    st.success(f"📊 Predicted Exam Score: **{predicted_score:.1f}**")
    st.markdown("---") 

    st.markdown("### 👥 Recommendations from Similar Students")
    
    # Default value untuk fitur yang tidak di model
    default_values_rec = {
        'gender': 1, 'part_time_job': 0, 'diet_quality': 2,
        'parental_education_level': 2, 'internet_quality': 2,
        'extracurricular_participation': 1,
        'age': 18
    }
    
    full_input_rec = {**user_inputs_for_model, **default_values_rec}
    input_df_rec = pd.DataFrame([full_input_rec])
    
    # Scale semua fitur untuk dibandingkan dengan user
    temp_scaler_rec = StandardScaler().fit(df_original.drop(columns=['exam_score']))
    input_scaled_rec = temp_scaler_rec.transform(input_df_rec[X_original_scaled_full.columns])

    # Euclidean distance antara input user dengan data lain
    distances = euclidean_distances(input_scaled_rec, X_original_scaled_full.values)

    # Indices dari 15 data paling mirip
    similar_student_indices = np.argsort(distances[0])[:15]
    neighbor_df = df_original.iloc[similar_student_indices]
    successful_neighbors = neighbor_df[neighbor_df['exam_score'] > 85]

    collaborative_recs = []
    if not successful_neighbors.empty:
        
        avg_neighbor_study = successful_neighbors['study_hours_per_day'].mean()
        if avg_neighbor_study > user_inputs_for_model['study_hours_per_day'] + 0.5:
            rec_text = (f"Students like you who scored over 85 studied for an "
                        f"average of **{avg_neighbor_study:.1f} hours/day**.")
            collaborative_recs.append(rec_text)
        
        avg_neighbor_sleep = successful_neighbors['sleep_hours'].mean()
        if avg_neighbor_sleep > user_inputs_for_model['sleep_hours'] + 0.5:
            rec_text = (f"Your successful peers got **{avg_neighbor_sleep:.1f} hours of sleep** on average.")
            collaborative_recs.append(rec_text)

        avg_neighbor_social = successful_neighbors['social_media_hours'].mean()
        if avg_neighbor_social < user_inputs_for_model['social_media_hours'] - 0.5:
            rec_text = (f"Your high-scoring peers only spent **{avg_neighbor_social:.1f} hours/day** on social media.")
            collaborative_recs.append(rec_text)

    if collaborative_recs:
        for rec in collaborative_recs:
            st.info(f"👥 **Peer Tip:** {rec}")
    else:
        st.info("Could not find a similar group of high-achieving students to offer tips.")