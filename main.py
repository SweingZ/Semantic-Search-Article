from typing import List
from fastapi import FastAPI, HTTPException
from opensearch_client import INDEX_NAME, connect_to_opensearch, index_articles, create_index
from embeddings import load_model, create_embeddings
from articles import load_articles_from_json
from models import SearchRequest, ArticleResponse

app = FastAPI()

# Initialize OpenSearch client and model
client = connect_to_opensearch()
model = load_model()
create_index(client)

# Load articles from JSON file and create embeddings
articles = load_articles_from_json("dummy_articles.json")
articles_with_embeddings = create_embeddings(articles, model)
index_articles(client, articles_with_embeddings)


@app.post("/search", response_model=List[ArticleResponse])
async def semantic_search(request: SearchRequest):
    try:
        query_embedding = model.encode(request.query).tolist()
        search_body = {
            "size": 3,  # Number of results to return
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_embedding,
                        "k": 3
                    }
                }
            }
        }
        response = client.search(index=INDEX_NAME, body=search_body)
        
        results = []
        for hit in response["hits"]["hits"]:
            result = ArticleResponse(
                title=hit["_source"]["title"],
                content=hit["_source"]["content"],
                score=hit["_score"]
            )
            results.append(result)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")
