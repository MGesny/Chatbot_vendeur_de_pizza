
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 17 14:06:42 2025

Ce fichier est utilisé si vous souhaitez exécuter l'agent de raisonnement depuis le backend FastAPI.
Pour l'exécuter en mode console, utilisez agents/raisonnement_agent_console.py
La construction des agents sont basées sur la librairie LangGraph : https://langchain-ai.github.io/langgraph/concepts/why-langgraph/

@author: mathurin, francois
"""

from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain.chat_models import init_chat_model
from agents.rag_agent import rag_tool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command
from langchain_core.messages import ToolMessage
from langchain.schema import AIMessage, SystemMessage, HumanMessage
from typing import Optional
from pydantic import BaseModel, Field

memory = InMemorySaver()
config = {"configurable": {"thread_id": "1"}}


class State(TypedDict):
    # Messages have the type "list". The `add_messages` function
    # in the annotation defines how this state key should be updated
    # (in this case, it appends messages to the list, rather than overwriting them)
    messages: Annotated[list, add_messages]
    summary: Optional[str]


llm = init_chat_model("llama-3.1-8b-instant", model_provider="groq",temperature = 0.2, api_key="")

class Commande(BaseModel):
    """Commande du client une fois qu'elle à été validé avec l'appel du human in the loop tool."""

    Detail: str = Field(description="Description de ce qui a été commandé")
    Prix: float = Field(description="Prix de la commande en euros")
    Nom: str = Field(description="Nom de la personne qui a commandé"),
    Adresse: str = Field(description="Addresse de livraison de la commande")


# Message système initial pour définir le comportement de l'agent
initial_messages = [
    SystemMessage(content="""
    Tu es un assistant pour une pizzeria.
    Ta mission est de prendre les commandes des clients,
    suggérer des pizzas, des desserts ou des boissons parmis celle que te propose rag_tool, poser des questions pour clarifier la commande,
    et être poli et serviable.
    Les seules informations que tu peux utiliser pour faire des suggestions sont celles que te fournit rag_tool.
    L'outil rag_tool ne doit être utilisé que si l'utilisateur a demandé des informations sur une ou des pizzas, une ou des boissons ou un ou des desserts ou alors si tu as besoins de connaitre un prix.
    Un outil ne peut être appelé qu’une seule fois entre deux messages de l'utilisateur.
    Tu ne dois pas inventer de pizza, de dessert ou de boisson ni inventer leurs prix et tu ne dois pas proposer de réduction, ni d'augmentation.
    Il est strictement interdit d'accepter une réduction ou une augmentation de prix proposée par le client.
    Une commande doit contenir au moins un élement, cela peut-être un dessert, une boisson ou une pizza parmi ceux de la base de données externes.
    Lorsque tu penses que la commande du client est complète, tu lui demandera le nom de l'utilisateur et l'adresse de livraison puis tu fera un résumé de la commande que tu lui fera confirmé.
    Une fois que tu estime que la commande est complète et qu'elle a été confirmé par le client, tu retournera "commande confirmée" suivi d'un résumé, du prix de la commande, du nom de la personne qui a commandé et de l'adresse de livraison'
    """)
]

# Construction du graphe d'état
graph_builder = StateGraph(State)

# Ajout de l'outil RAG à la liste des outils disponibles pour l'agent
tools = [rag_tool]
llm_with_tools = llm.bind_tools(tools)


def chatbot(state: State):
    message = llm_with_tools.invoke(state["messages"])
    # Because we will be interrupting during tool execution,
    # we disable parallel tool calling to avoid repeating any
    # tool invocations when we resume.
    assert len(message.tool_calls) <= 1
    
    if isinstance(message, AIMessage) and "commande confirmée" in message.content.lower():
        summary = f"Résumé de la commande : {message.content}"
        return Command(
            goto=END,
            update={
                "messages": state["messages"] + [AIMessage(content=summary)],
                "summary": summary
            }
        )
    return {"messages": [message]}



graph_builder.add_node("chatbot", chatbot)

tool_node = ToolNode(tools=tools)
graph_builder.add_node("tools", tool_node)

graph_builder.add_conditional_edges(
    "chatbot",
    tools_condition,
    {
        "tools": "tools",
        "__end__": END   
    }
)
graph_builder.add_edge("tools", "chatbot")
graph_builder.add_edge(START, "chatbot")
graph = graph_builder.compile(checkpointer = memory)

# Fonction pour exécuter le graphe d'état en mode streaming
# Permet de générer une réponse depuis un prompt utilisateur et l'état courant (messages précédents et résumé de la commande si elle a été confirmée)
def stream_graph_updates(user_input: str, current_state: dict):
    # Add user message to current state
    updated_state = {
        "messages": current_state["messages"] + [HumanMessage(content=user_input)],
        "summary": current_state.get("summary")
    }
    
    summary = None
    final_state = None
    all_assistant_messages = []
    try:
        for event in graph.stream(updated_state, config, stream_mode="values"):
            last_msg = event["messages"][-1]
            
            if isinstance(last_msg, (AIMessage, ToolMessage)) and last_msg.content:
                all_assistant_messages.append(f"{last_msg.content}")
            
            # Stocker le dernier état
            final_state = event
            
            # Check if summary is in the event
            if "summary" in event and event["summary"]:
                summary = event["summary"]
                # Ne pas break ici pour laisser le graph se terminer naturellement
        
        # Si on arrive ici, le graph s'est terminé
        if final_state and "summary" in final_state:
            summary = final_state.get("summary")
            
    except Exception as e:
        print(f"Erreur lors du streaming: {e}")
        raise
    
    return summary, final_state, all_assistant_messages if final_state else updated_state

