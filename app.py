import chainlit as cl
from pymilvus import MilvusClient
import requests
from vector_db import query_hybrid_search
from ollama import chat

message_history = []

@cl.on_message
async def handle_message(message: cl.Message):
    query = message.content

    # Append the new message to the history
    message_history.append({"role": "user", "content": query})

    # retrieve relevant documents from the vector database
    # docs = query_hybrid_search(query)
    # context = "\n".join(docs)
    # context = docs

    # Append the context to the message history
    # message_history.append({"role": "system", "content": context})

    response = chat(
        model='gemma3:4b-it-qat',
        messages=message_history
    )

    assistant_reply = response['message']['content']

    # Append the assistant's response to the message history
    message_history.append({"role": "assistant", "content": assistant_reply})

    await cl.Message(content=assistant_reply).send()
