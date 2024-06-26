from langchain import FAISS
import pandas as pd

# Load dataset
animal_data = pd.read_csv("animal-fun-facts-dataset.csv")

from langchain.embeddings import SentenceTransformerEmbeddings

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

metadatas = []
for i, row in animal_data.iterrows():
    metadatas.append(
        {
            "Animal Name": row["animal_name"],
            "Source URL": row["source"],
            # "Media URL": row["media_link"],
            # "Wikipedia URL": row["wikipedia_link"],
        }
    )

animal_data["text"] = animal_data["text"].astype(str)

faiss = FAISS.from_texts(animal_data["text"].to_list(), embedding_function, metadatas)

faiss.similarity_search_with_score("What is ship of the desert?", 3)

#export vector-db
faiss.save_local("faiss_db", "animal-fun-facts")

from langchain_community.vectorstores import FAISS

#load vector-db
prebuilt_faiss=FAISS.load_local("faiss_db", embedding_function, "animal-fun-facts",allow_dangerous_deserialization=True)

