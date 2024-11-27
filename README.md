This project aims to build a fake news detection system using machine learning to classify 
news articles as real or fake. The system leverages a labeled dataset of news articles with two 
categories: "REAL" and "FAKE." To preprocess the data, missing values are handled, and labels 
are encoded numerically for compatibility with machine learning models.
The text data is vectorized using TF-IDF (Term Frequency-Inverse Document Frequency), 
a technique that transforms the raw text into numerical features by considering word frequency, 
helping distinguish between key terms in fake and real news. With these TF-IDF vectors, four 
popular classification models are trained and evaluated: Logistic Regression, Naive Bayes, 
Support Vector Machine (SVM), and Random Forest.
Each model’s performance is measured by accuracy, allowing us to identify the model that 
best separates fake news from real news. The approach provides an efficient way to automatically 
assess news credibility, helping users and platforms combat the spread of misinformation.
