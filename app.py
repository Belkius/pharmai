import chainlit as cl
from pymilvus import MilvusClient
import requests
from vector_db import query_hybrid_search
from ollama import chat

message_history = []

# @cl.on_message
# async def handle_message(message: cl.Message):
#     query = message.content

#     # Append the new message to the history
#     message_history.append({"role": "user", "content": query})

#     # retrieve relevant documents from the vector database
#     docs = query_hybrid_search(query)
#     # context = "\n".join(docs)
#     context = docs[0][0]['text']
#     # halfing the context to 1000 characters
#     # if len(context) > 1000:
#     #     context = context[:10000]
#     # print(f"Context: {context}")

#     # Append the context to the message history
#     message_history.append({"role": "system", "content": context})

#     response = chat(
#         # model='gemma3:4b-it-qat',
#         # model='gemma-4b-128k',
#         model='hf.co/unsloth/gemma-3-4b-it-GGUF:Q4_K_M',
#         messages=message_history,
#         options={
#             "num_ctx": 128000,
#         }
#     )

#     assistant_reply = response['message']['content']

#     # Append the assistant's response to the message history
#     message_history.append({"role": "assistant", "content": assistant_reply})
#     print(message_history)

#     await cl.Message(content=assistant_reply).send()

    
# @cl.on_message
# async def handle_message(message: cl.Message):
#     query = message.content

#     # Step 1: Ask the model if it needs context
#     check_response = chat(
#         model='gemma3:4b-it-qat',
#         messages=[
#             {"role": "system", "content": "You are a helpful assistant. Only request extra data if necessary."},
#             {"role": "user", "content": f"Do you need additional documents from a medical products characteristics database to answer this question: '{query}'? Reply 'yes' or 'no' only."}
#         ],
#         options={
#             "num_ctx": 128000,
#         }
#     )

#     print(f"Check response: {check_response}")
    
#     needs_rag = check_response['message']['content'].strip().lower()

#     # Step 2: Conditional RAG
#     if "yes" in needs_rag:
#         docs = query_hybrid_search(query)
#         context = docs[0][0]['text']
#         message_history = [
#             {"role": "system", "content": f"Use this context to answer: {context}"},
#             {"role": "user", "content": query}
#         ]
#     else:
#         message_history = [{"role": "user", "content": query}]

#     # Final answer
#     final_response = chat(model='gemma3:4b-it-qat', messages=message_history, 
#         options={
#             "num_ctx": 128000
#         }
#     )
#     assistant_reply = final_response['message']['content']

#     await cl.Message(content=assistant_reply).send()

