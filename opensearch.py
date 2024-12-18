from opensearchpy import OpenSearch
from sentence_transformers import SentenceTransformer
import json

INDEX_NAME = "articles"

# Function to connect to OpenSearch
def connect_to_opensearch(host='localhost', port=9200):
    client = OpenSearch(
        hosts=[{'host': host, 'port': port}],
        use_ssl=False,
        verify_certs=False
    )
    if client.ping():
        print("Connected to OpenSearch!")
        return client
    else:
        print("Failed to connect to OpenSearch.")
        raise ConnectionError("Failed to connect to OpenSearch.")


# Function to create an OpenSearch index
def create_index(client, index_name = INDEX_NAME):
    if client.indices.exists(index=index_name):
        print(f"Index {index_name} already exists. Deleting and recreating it.")
        client.indices.delete(index=index_name)

    # Define index mapping
    mapping = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "index": {
                "knn": True  
            }
        },
        "mappings": {
            "properties": {
                "title": {"type": "text"},
                "content": {"type": "text"},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": 384  
                }
            }
        }
    }

    # Create the index
    response = client.indices.create(index=index_name, body=mapping)
    print(f"Index {index_name} created: {response}")
    return index_name


# Function to create embeddings for articles
def create_embeddings(articles, model):
    for article in articles:
        text = f"{article['title']} {article['content']}"  
        article['embedding'] = model.encode(text).tolist() 
    print("Embeddings created.")
    return articles


# Function to index articles with embeddings in OpenSearch
def index_articles(client, articles, index_name= INDEX_NAME):
    for article in articles:
        doc = {
            "title": article['title'],
            "content": article['content'],
            "embedding": article['embedding']
        }
        response = client.index(index=index_name, body=doc)
        print(f"Indexed document ID: {response['_id']}")


# Function to perform a semantic search query
def semantic_search(client, query, model, index_name= INDEX_NAME, k=3):
    query_embedding = model.encode(query).tolist()
    print(f"Query embedding length: {len(query_embedding)}")

    # Semantic search query
    search_body = {
        "size": k,  
        "query": {
            "knn": {
                "embedding": {
                    "vector": query_embedding,
                    "k": k  
                }
            }
        }
    }

    try:
        response = client.search(index=index_name, body=search_body)
        print("Search results:")
        for hit in response["hits"]["hits"]:
            print(f"Title: {hit['_source']['title']}")
            print(f"Content: {hit['_source']['content']}")
            print(f"Score: {hit['_score']}")
            print("------")
    except Exception as e:
        print(f"Error during search: {e}")


# Function to load articles from JSON file
def load_articles_from_json(file_path):
    with open(file_path, 'r') as file:
        articles = json.load(file)
    return articles


def main():
    client = connect_to_opensearch()

    index_name = create_index(client)

    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # Example articles
    articles = load_articles_from_json("dummy_articles.json")

    articles_with_embeddings = create_embeddings(articles, model)

    index_articles(client, articles_with_embeddings, index_name)

    query = "How is AI used in healthcare industry?"
    semantic_search(client, query, model, index_name)


if __name__ == "__main__":
    main()
