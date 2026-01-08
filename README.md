# 🏆 AutoJudge Pro: CP Problem Predictor

## 1. Project Overview
**AutoJudge Pro** is an intelligent machine learning tool designed for Competitive Programming (CP) enthusiasts. It analyzes the text description of a coding problem and automatically predicts:
* **Difficulty Level:** (Easy, Medium, or Hard)
* **Estimated Rating:** (e.g., 800, 1400, 2100 similar to Codeforces ratings)

This tool helps problem setters automatically tag their problems and helps students gauge the difficulty of unlabelled questions.

## 2. Dataset Used
* **Source:** Custom `problems.jsonl` dataset containing competitive programming problems.
* **Format:** JSON Lines (JSONL).
* **Features Used:** `description`, `input_description`, `output_description`.
* **Target Variables:** `problem_class` (Classification) and `problem_score` (Regression).

## 3. Approach and Models
We used a **Supervised Machine Learning** approach with the following pipeline:

### Data Preprocessing:
* Text cleaning (HTML tag removal, lowercase conversion).
* Concatenation of problem description, input, and output text.

### Feature Extraction:
* **TF-IDF Vectorization:** To convert text into numerical vectors (Top 1000 features).
* **Text Length Feature:** Added the length of the problem statement as a normalized feature (Harder problems tend to be longer).

### Models:
* **Classification (Difficulty):** `RandomForestClassifier` (n_estimators=100).
* **Regression (Rating):** `RandomForestRegressor` (n_estimators=100).

## 4. Evaluation Metrics
To ensure model reliability, the following metrics were achieved on the test set:

* **Classification Accuracy:** **55%** (0.55)
    * *Measures how often the model correctly predicts Easy/Medium/Hard.*
* **Regression Mean Absolute Error (MAE):** **467.39**
    * *On average, the predicted rating deviates by ~467 points.*
* **Regression Root Mean Squared Error (RMSE):** **547.00**
    * *Indicates the model is stable with few extreme outliers.*

## 5. Steps to Run Locally
Follow these steps to run the project on your own machine:

1.  **Prerequisites:** Python 3.8+ installed.
2.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/prathamesh2705558/AutoJudge_Project.git](https://github.com/prathamesh2705558/AutoJudge_Project.git)
    cd AutoJudge_Project
    ```
3.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
4.  **Train the Models (Optional):**
    *(Pre-trained models are included, but to retrain:)*
    ```bash
    python train_model.py
    ```
5.  **Run the Web App:**
    ```bash
    streamlit run app.py
    ```

## 6. Explanation of Web Interface
The User Interface is built using **Streamlit** for a clean, responsive experience:
* **Input Area:** Three text boxes for Problem Description, Input Format, and Output Format.
* **Analyze Button:** Triggers the ML pipeline to process the text.
* **Output Dashboard:**
    * **Difficulty Label:** Displays Easy (Green), Medium (Gold), or Hard (Red).
    * **Predicted Rating:** Shows the specific numeric rating with a visual progress bar.

## 7. Demo Video
Watch the working demo of the project here: https://youtu.be/v_9dLJEyRNU

## 8. Author Details
* **Name:** Prathamesh Amrutkar
* **GitHub:** [prathamesh2705558](https://github.com/prathamesh2705558)
* **Project Type:** Machine Learning / NLP
