from sentence_transformers import SentenceTransformer
import pymongo

mongo_uri = "mongodb+srv://xxx:xxx@cluster-xyz.mongodb.net/?retryWrites=true&w=majority"
db = "sample_mflix"
collection = "movies"

# initialize db connection
connection = pymongo.MongoClient(mongo_uri)
collection = connection[db][collection]

# define transofrmer model (from https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

for x in collection.find({'vector': {'$exists': False}}, {}):
    # checking if vector already computed for this doc
    if('vector' not in x.keys()):
        if('title' in x.keys()):
            movieid = x['_id']
            title = x['title']
            print('computing vector.. title: ' + title)
            text = title
            
            # if fullpplot field present, concat it with title
            if('fullplot' in x.keys()):
                fullplot = x['fullplot']
                text = text + ". " + fullplot

            vector = model.encode(text).tolist()

            collection.update_one({'_id': movieid},{ "$set": { 'vector': vector, 'title': title, 'fullplot': fullplot } }, upsert = True)
            print('vector computed!!')
    else:
        print('vector already computed')
