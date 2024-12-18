from sentence_transformers import SentenceTransformer

def create_embeddings(articles, model):
    for article in articles:
        text = f"{article['title']} {article['content']}"
        article['embedding'] = model.encode(text).tolist()
    return articles

def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
