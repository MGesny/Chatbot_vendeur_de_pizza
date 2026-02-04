# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 23:01:04 2025

Ce fichier définit l'outil RAG (Retrieval-Augmented Generation) utilisé par l'agent de raisonnement.
La construction des agents sont basées sur la librairie LangGraph : https://langchain-ai.github.io/langgraph/concepts/why-langgraph/

@author: mathurin, francois
"""

from langchain.chat_models import init_chat_model
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain.tools import tool
from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma

# Initialise le LLM et la base de données vectorielle
llm = init_chat_model("llama-3.1-8b-instant", model_provider="groq",temperature=0, api_key="")

embeddings = OllamaEmbeddings(model="granite-embedding:278m")

vector_store = Chroma(
    persist_directory="./db/db_rag_pizzeria",
    embedding_function=embeddings
)

# Sert à définir le comportement de l'agent RAG
prompt = PromptTemplate(
    input_variables=["context", "question"],
    template="""Tu es un assistant utile.

Utilise uniquement le contexte suivant pour répondre à la question

Contexte:
{context}

Question:
{question}

Répond aussi clairement et aussi précisemment que possible si tu ne sais pas, n'invente pas.
""")


# Définition du type d'état pour le graphe
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str


# Défini les étapes du graphe
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"],k=40) # Nous avons choisi de retourner 40 sous-parties de documents comme il semblait que c'était ce qui marchait le mieux en prenant en compte la taille du contexte utilisable dans le LLM
    return {"context": retrieved_docs}

# Génération de la réponse basée sur le contexte récupéré
def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}


# Construction du graphe d'état
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

# Définition de comment l'outil RAG va être utilisé par l'agent de raisonnement
@tool
def rag_tool(question: str) -> str:
    """Utilise cet outil quand tu as besoin de répondre à une question sur les pizzas, sur les desserts ou sur les boissons basée sur un contexte externe (retrouvé dans les documents).
    Le paramètre doit toujours s’appeler 'question'."""
    response = graph.invoke({"question": question})
    return response["answer"]



