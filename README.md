# 🎓 Student Performance Predictor & Peer Recommendation System

A web application built with Streamlit that predicts a student's exam score based on their daily habits and provides personalized, data-driven recommendations for improvement by comparing them to similar, high-achieving peers.

This project demonstrates a complete machine learning workflow, from data exploration and model training in a Jupyter Notebook to deploying a predictive model in an interactive web app.

# ✨ Features

- 📈 Score Prediction: Utilizes a trained Linear Regression model to predict a student's potential exam score based on seven key lifestyle and academic habits.
- 👥 Peer-Based Recommendations: Implements a collaborative filtering-style recommendation engine. It identifies a cohort of high-performing students (exam score > 85) with habits most similar to the user (using Euclidean distance) and offers actionable tips based on their collective behavior.
- 🎚️ Interactive UI: A simple and intuitive user interface created with Streamlit, using sliders for easy input of user data.

# 🛠️ Tech Stack

- Backend & Machine Learning: Python, Scikit-learn, Pandas, NumPy, Joblib
- Frontend: Streamlit
- Data Analysis & Visualization: Jupyter Notebook, Matplotlib, Seaborn

# 📂 Project Structure

```bash
.
├── App.py                      # The main Streamlit web application file
├── Main.ipynb                  # Jupyter Notebook for EDA, model training, and evaluation
├── model.pkl                   # Serialized final Linear Regression model
├── scaler.pkl                  # Serialized StandardScaler for the 7 selected features
├── label_encoders.pkl          # Serialized LabelEncoders for categorical data transformation
├── student_habits_performance.csv # The dataset used for training
├── requirements.txt            # List of Python dependencies
└── README.md                   # You are here!
```

🔧 Setup and Installation

To run this project locally, please follow these steps:

1. Clone the repository:

  ```bash
  git clone https://github.com/your-username/student-performance-predictor.git
  cd student-performance-predictor
  ```
2. Create and activate a virtual environment:

  On macOS/Linux:
  ```bash
  python3 -m venv venv
  source venv/bin/activate
  ```
  On Windows:
  ```bash
  python -m venv venv
  .\venv\Scripts\activate
  ```
3. Install the required dependencies:
A requirements.txt file is included for easy installation.
  ```bash
  pip install -r requirements.txt
  ```

If you don't have the requirements.txt file, you can create one with the following content:
```bash
  streamlit
  pandas
  scikit-learn
  numpy
  joblib
```
# 🚀 How to Run

Once the setup is complete, you can launch the Streamlit application with the following command:
  ```bash
  streamlit run App.py
  ```

Your web browser should automatically open to the application's local URL.

🧠 Methodology

The predictive model was developed following these steps, as detailed in Main.ipynb:

1. Data Exploration (EDA): The student_habits_performance.csv dataset was analyzed to understand feature distributions, identify outliers (using boxplots), and examine correlations between variables (using a heatmap).

2. Preprocessing:

  - Categorical features (gender, part_time_job, etc.) were converted to numerical format using LabelEncoder.
  - All numerical features were standardized using StandardScaler to ensure they were on a comparable scale for model training.

3. Model Training and Tuning: Four different regression models were evaluated:

  - Linear Regression
  - Support Vector Regressor (SVR)
  - Gradient Boosting Regressor
  - Random Forest Regressor

4. GridSearchCV was used to perform hyperparameter tuning and identify the best-performing version of each model based on the R² score.

5. Feature Selection: After comparing the tuned models, Linear Regression was selected as the final model due to its high R² score (0.899) and excellent interpretability. The absolute values of its coefficients were used to determine feature importance. The top 7 features were selected for the final model:

  - study_hours_per_day
  - mental_health_rating
  - social_media_hours
  - exercise_frequency
  - sleep_hours

        netflix_hours

        attendance_percentage

    Final Model Creation: The Linear Regression model was retrained using only these top 7 features. The final model (model.pkl) and its corresponding scaler (scaler.pkl) were saved for use in the Streamlit application.
