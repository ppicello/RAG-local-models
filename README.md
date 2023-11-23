# RAG (Retrieval Augmented Generation) with local models

![header](/docs/header.png?raw=true "header")

This repo wants to be an easy way to showcase how to implement RAG (Retrieval Augemented Generation) with only local models (no need for OpenAI api keys).
For this example we leverage MongoDB [Atlas Search](https://www.mongodb.com/docs/atlas/atlas-search/) as Vector Store, [Hugging Face](https://huggingface.co/) transformers to compute the embeddings, [Langchain](https://python.langchain.com/docs/get_started/introduction) as LLM framework and a quantized version of Llama2-7B from [TheBloke's](https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF).

> **Note**
> You are free to experiment with different models, the main goal of this work was to get something that can run entirely on CPU with the least amount of memory possible. Bigger models will give you way better results. 


<a id="AtlasCluster"></a>

## Load sample data

In this example we are using the `sample_mflix.movies` collection, part of the sample dataset available in Atlas. 
Once you have uploaded the sample dataset in you Atlas Cluster, you can run encoder.py to compute vectors out of your MongoDB documents. 


```console
python3 encoder.py
```


## Define Atlas Search Index

Create the following search index on the `sample_mflix.movies` collection:

```json
{
  "mappings": {
    "fields": {
      "vector": [
        {
          "dimensions": 384,
          "similarity": "cosine",
          "type": "knnVector"
        }
      ]
    }
  }
}
```


<a id="test1"></a>
## RAG: Example 1

In this first example we are going to ask our LLM to give suggestions about movies. The app will first compute the vector from the user query, this vector will be used by Atlas Search to identify the most relevant results. These results are then sent to the LLM for the answer generation.

> Give me some movies about race cars

![](/docs/test-rag.gif?raw=true)

As you can see the system is using the information coming from the Vector Store to answer the user question.

<a id="test2"></a>
## RAG: Example 2

In this second example we are going to first of all insert a fake document in our collection, compute the embeddings, and then we are going to ask a question about this movie to prove that our LLM is actually using the information coming from the Vector Store. 

This is the fake movie document: 
```json
{
	"fullplot": "The Fake movie. This fictitious movie was created by Paolo Picello, an italian computer engineer. Paolo is trying to build an AI that can answer questions around popular movies and is trying to do so with MongoDB Atlas, Langchain and Llama 2, an open source large language.",
	"title": "The Fake Movie"
}
```

We can then ask the LLM something like: 

> What is the name of the movie where Paolo Picello is trying to build an AI to answer questions?

![](/docs/test-rag-fake-movie.gif?raw=true)

As you can see the system is able to give me the correct name of the movie, even if this movie does not really exist (it was not in the training data for the LLM).
