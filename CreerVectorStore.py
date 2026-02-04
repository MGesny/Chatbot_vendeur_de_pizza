# -*- coding: utf-8 -*-
"""
Created on Tue Sep 16 20:43:46 2025

@author: mathurin, francois
"""

from langchain_ollama import OllamaEmbeddings
from langchain.document_loaders import DirectoryLoader, TextLoader, CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
import os
import pandas as pd
from langchain.schema import Document

# Charger le CSV avec pandas
df = pd.read_csv("./PrixPizzeria.csv", delimiter=";")

csv_docs = []


for i, row in df.iterrows():
    # Construire un texte clair, par exemple : "Nom: Pizza Reine, Prix: 12.50, Taille: Grande"
    content = ", ".join([f"{col}: {row[col]}" for col in df.columns]) # Plus performant que le parser CSV de langchain
    
    doc = Document(
        page_content=content,
        metadata={"source": "PrixPizzeria.csv", "row": i}
    )
    csv_docs.append(doc)
# Liste des pizzas disponibles    
doc = Document(
    page_content="Les pizzas disponibles sont " +", ".join(df[df.Categorie == "Pizza"].Nom.unique())+" en format large ou moyen."
    )
csv_docs.append(doc)
# Desserts
doc = Document(
    page_content="Les desserts disponibles sont " +", ".join(df[df.Categorie == "Dessert"].Nom.unique())+"."
    )
csv_docs.append(doc)
#Boissons
doc = Document(
    page_content="Les boissons disponibles sont " +", ".join(df[df.Categorie == "Boissons"].Nom.unique())+"."
    )
csv_docs.append(doc)

# Initialisation du modèle d'embedding
embeddings = OllamaEmbeddings(model="granite-embedding:278m") # modèle multilingue 

# Load les différentes recettes
loader = DirectoryLoader(
    "./Recettes",
    glob="*.txt",
    loader_cls=TextLoader
)
docs = loader.load()

# Découpe les documents textuelles en chunk de taille 500 tokens qui se chevauche de 100 tokens
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100) # Meilleure tentative que nous avons faites (mais il est peut-être possible de trouver mieux)

csv_splits = text_splitter.split_documents(csv_docs)
text_splits = text_splitter.split_documents(docs)

# Enregistrement des textes et des embeddings dans une base chromadb (pour la simplicité)
vectordb = Chroma.from_documents(
    text_splits+csv_splits,
    embeddings,
    persist_directory="./db_rag_pizzeria"
)
vectordb.persist()
