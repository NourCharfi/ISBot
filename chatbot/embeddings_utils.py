from chatbot.data_processing import fasttext_model, get_document_vector_fasttext, fasttext_question_vectors
import numpy as np

def get_best_match_with_fasttext(user_input):
    """
    Find the best matching question using FastText embeddings.
    Returns (best_index, similarity_score)
    """
    if fasttext_model is None or fasttext_question_vectors is None:
        return 0, 0.0
    input_vec = get_document_vector_fasttext(user_input, fasttext_model)
    similarities = np.dot(fasttext_question_vectors, input_vec) / (
        np.linalg.norm(fasttext_question_vectors, axis=1) * np.linalg.norm(input_vec) + 1e-8)
    best_idx = int(np.argmax(similarities))
    return best_idx, float(similarities[best_idx]) 