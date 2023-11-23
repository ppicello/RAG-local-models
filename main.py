from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.llms import LlamaCpp

from langchain.vectorstores import MongoDBAtlasVectorSearch
from langchain.embeddings import HuggingFaceEmbeddings
from pymongo import MongoClient

uri="mongodb+srv://xxx:xxx@cluster-xyz.mongodb.net/?retryWrites=true&w=majority"
client = MongoClient(uri)

db_name = "sample_mflix"
collection_name = "movies"
collection = client[db_name][collection_name]

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
index_name = "vector_search_index"
vector_field_name = "vector"
text_field_name = "title"

vectorStore = MongoDBAtlasVectorSearch(collection=collection,embedding=embeddings, index_name=index_name, embedding_key=vector_field_name, text_key=text_field_name)
query = "Give me some movies about race cars"
#query = "What is the name of the movie where Paolo Picello is trying to build an AI to answer questions?"

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm = LlamaCpp(
    model_path="models/llama-2-7b-chat.Q2_K.gguf",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

retriever = vectorStore.as_retriever(search_kwargs={"k": 1})

print("\n")
print("\n")
print("User Query: \"" + query + "\"")
print("\n")

qa = RetrievalQA.from_chain_type(llm, chain_type="stuff", retriever=retriever)

# Execute the chain
qa.run(query)

print("\n")
print("\n")
print("\n")
print("\n")

# Example output
#  1) I'm glad you asked! Here are some movies about race cars that might interest you:
# 1. "Sleeping Beauty" (1953) - This classic film features a race between two drivers, one of whom is the legendary driver Jeff Gordon.
# 2. "The Fast and the Furious" (2001) - This action-packed movie follows a group of street racers as they participate in high-stakes races across the country.
# 3. "Talladega Nights: The Ballad of Ricky Bobby" (2006) - In this comedy, Ricky Bobby (played by Will Ferrell) becomes a NASCAR sensation and must navigate the cutthroat world of professional racing.
# 4. "Rush" (2013) - This biographical drama tells the story of Formula One race car drivers James Hunt and Niki Lauda, who competed against each other in the 1970s.
# 5. "Drive" (2011) - Set in the gritty streets of 1980s Los Angeles, this indie drama follows a young stunt driver (played by Ryan Gosling) as he gets caught up in the dangerous world of underground street racing.
#
#  2) The name of the movie is "The Fake Movie"
