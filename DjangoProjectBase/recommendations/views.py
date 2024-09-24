from django.shortcuts import render
from movie.models import Movie
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import json

load_dotenv(find_dotenv('api_keys.env'))


def get_embedding(text, client, model="text-embedding-3-small"):
   text = text.replace("\n", " ")
   return client.embeddings.create(input = [text], model=model).data[0].embedding

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Create your views here.
def recommendations(request):
    #Se lee del archivo .env la api key de openai
    req = request.GET.get('search')
    if req:
        client = OpenAI(
        # This is the default and can be omitted
            api_key=os.getenv('openai_api_key'),
        )
        
        items = Movie.objects.all()
        emb_req = get_embedding(req, client)

        sim = []
        for i in range(len(items)):
            emb = items[i].emb
            emb = list(np.frombuffer(emb))
            sim.append(cosine_similarity(emb,emb_req))
        sim = np.array(sim)
        idx = np.argmax(sim)
        idx = int(idx)
        movies = [items[idx]]
    else:
        movies = Movie.objects.all()

    return render(request, 'recommendations.html', {'movies':movies, 'search':req})