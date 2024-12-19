from opensearchpy import OpenSearch

INDEX_NAME = "articles"

def connect_to_opensearch():
    client = OpenSearch(
        hosts=[{'host': 'localhost', 'port': 9200}],
        use_ssl=False,
        verify_certs=False
    )
    if client.ping():
        print("Connected to Opensearch")
        return client
    else:
        raise ConnectionError("Failed to connect to OpenSearch.")

def create_index(client):
    if client.indices.exists(index=INDEX_NAME):
        client.indices.delete(index=INDEX_NAME)
    
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

    response = client.indices.create(index=INDEX_NAME, body=mapping)
    return INDEX_NAME

def index_articles(client, articles):
    for article in articles:
        doc = {
            "title": article['title'],
            "content": article['content'],
            "embedding": article['embedding']
        }
        client.index(index=INDEX_NAME, body=doc)
