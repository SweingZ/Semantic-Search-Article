from random import sample
from typing import List
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from opensearch_client import INDEX_NAME, connect_to_opensearch, index_articles, create_index
from embeddings import load_model, create_embeddings
from articles import load_articles_from_json
from models import SearchRequest, ArticleResponse

app = FastAPI()

client = connect_to_opensearch()
model = load_model()
create_index(client)

# Load articles from JSON file and create embeddings
articles = load_articles_from_json("dummy_articles.json")
articles_with_embeddings = create_embeddings(articles, model)
index_articles(client, articles_with_embeddings)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@app.post("/good-search", response_model=List[ArticleResponse])
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
                author=hit["_source"]["author"],
                published_date=hit["_source"]["published_date"],
                score=hit["_score"]
            )
            results.append(result)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during search: {str(e)}")

@app.post("/bad-search", response_model=List[ArticleResponse])
async def bad_search(request: SearchRequest):
    try:
        search_body = {
            "size": 3, 
            "query": {
                "match": {
                    "content": request.query
                }
            }
        }
        response = client.search(index=INDEX_NAME, body=search_body)

        results = []
        for hit in response["hits"]["hits"]:
            result = ArticleResponse(
                title=hit["_source"]["title"],
                content=hit["_source"]["content"],
                author=hit["_source"]["author"],
                published_date=hit["_source"]["published_date"],
                score=hit["_score"]
            )
            results.append(result)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during bad search: {str(e)}")


    
@app.get("/random-articles", response_model=List[ArticleResponse])
async def get_random_articles():
    try:
        search_body = {
            "size": 1000,  # Fetch a large number of documents to sample from
            "query": {
                "match_all": {}
            }
        }
        response = client.search(index=INDEX_NAME, body=search_body)
        
        # Sample 6 random articles
        total_articles = response["hits"]["hits"]
        if len(total_articles) < 9:
            raise HTTPException(status_code=404, detail="Not enough articles to fetch 9 random ones.")
        
        random_articles = sample(total_articles, 9)
        
        results = []
        for hit in random_articles:
            result = ArticleResponse(
                title=hit["_source"]["title"],
                content=hit["_source"]["content"],
                author=hit["_source"]["author"],
                published_date=hit["_source"]["published_date"],
                score=hit["_score"]
            )
            results.append(result)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching random articles: {str(e)}")
    
@app.get("/recommend-articles", response_model=List[ArticleResponse])
async def recommend_articles_by_title(title: str):
    try:
        # Step 1: Fetch the article by title
        search_body = {
            "query": {
                "match": {
                    "title": title
                }
            }
        }
        response = client.search(index=INDEX_NAME, body=search_body)
        
        if not response["hits"]["hits"]:
            raise HTTPException(status_code=404, detail="Article not found.")
        
        # Get the ID and embedding of the matched article
        original_article = response["hits"]["hits"][0]
        article_id = original_article["_id"]
        article_embedding = original_article["_source"]["embedding"]

        # Step 2: Use the embedding to find the next 3 closest articles, excluding the original one
        recommendation_body = {
            "size": 4,  
            "query": {
                "bool": {
                    "must": {
                        "knn": {
                            "embedding": {
                                "vector": article_embedding,
                                "k": 4  
                            }
                        }
                    },
                    "must_not": {
                        "term": {
                            "_id": article_id  
                        }
                    }
                }
            }
        }
        rec_response = client.search(index=INDEX_NAME, body=recommendation_body)

        # Step 3: Extract the results
        results = []
        for hit in rec_response["hits"]["hits"][:3]:  # Only return the next 3 closest
            result = ArticleResponse(
                title=hit["_source"]["title"],
                content=hit["_source"]["content"],
                author=hit["_source"]["author"],
                published_date=hit["_source"]["published_date"],
                score=hit["_score"]
            )
            results.append(result)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during recommendation: {str(e)}")

