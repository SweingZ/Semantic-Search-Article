import json

def load_articles_from_json(file_path):
    with open(file_path, 'r') as file:
        return json.load(file)
