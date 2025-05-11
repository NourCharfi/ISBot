import os
import pickle
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from chatbot.data_processing import processed_questions, categories, vectorizer, tfidf_matrix

# Split data
X_train, X_test, y_train, y_test = train_test_split(processed_questions, categories, test_size=0.2, random_state=42)
X_train_tfidf = vectorizer.transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train Naive Bayes classifier
nb_classifier = MultinomialNB(alpha=0.1)
nb_classifier.fit(X_train_tfidf, y_train)

# Train KNN classifier
X_train_dense = X_train_tfidf.toarray()
knn_classifier = KNeighborsClassifier(n_neighbors=5, metric='cosine')
knn_classifier.fit(X_train_dense, y_train)

# Save models
if not os.path.exists("models"):
    os.makedirs("models")
with open('models/nb_classifier.pkl', 'wb') as f:
    pickle.dump(nb_classifier, f)
with open('models/knn_classifier.pkl', 'wb') as f:
    pickle.dump(knn_classifier, f)
with open('models/vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)