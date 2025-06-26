from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, END
from langchain.schema import Document
from typing_extensions import TypedDict
from typing import List
import json
from google import genai
from google.genai import types
from langchain_google_genai import ChatGoogleGenerativeAI


from vector_db import query_hybrid_search, extract_ids, load_texts_from_ids, headers_extraction
from langchain_community.tools import DuckDuckGoSearchResults

### --- STATE DEFINITION ---
class GraphState(TypedDict):
    question: str
    generation: str
    web_search: str
    documents: List[str]
    relevant_headers: List[str]


### --- MODEL INIT ---
model_type = 'google'  # 'local' or 'google'

def get_llm(model_type: str = 'local'):
    """
    Returns a language model instance based on the specified model_type.

    Parameters:
    - model_type (str): Type of the model to use. Options are 'local' or 'google'.

    Returns:
    - An instance of the selected language model.
    """
    if model_type == 'local':
        return ChatOllama(
            model='hf.co/unsloth/gemma-3-4b-it-GGUF:Q4_K_M',
            format="json"
        )
    elif model_type == 'google':
        with open('secrets.json') as f:
            secrets = json.load(f)
            GOOGLE_API_KEY = secrets['GOOGLE_API_KEY']
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=GOOGLE_API_KEY
        )
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")


### --- TOOL ---
# web_search_tool = DuckDuckGoSearchResults(region="pl-pl", max_results=3)
web_search_tool = DuckDuckGoSearchResults()

headers = """NAZWA PRODUKTU LECZNICZEGO, SKŁAD JAKOŚCIOWY I ILOŚCIOWY, POSTAĆ FARMACEUTYCZNA, SZCZEGÓŁOWE DANE KLINICZNE, WŁAŚCIWOŚCI FARMAKOLOGICZNE, DANE FARMACEUTYCZNE, PODMIOT ODPOWIEDZIALNY POSIADAJĄCY POZWOLENIE NA DOPUSZCZENIE DO OBROTU, NUMER POZWOLENIA NA DOPUSZCZENIE DO OBROTU, DATA WYDANIA PIERWSZEGO POZWOLENIA NA DOPUSZCZENIE DO OBROTU I DATA PRZEDŁUŻENIA POZWOLENIA, DATA ZATWIERDZENIA LUB CZĘŚCIOWEJ ZMIANY TEKSTU CHARAKTERYSTYKI PRODUKTU LECZNICZEGO"""


# ### --- CHAINS ---
llm = get_llm(model_type=model_type)
retrieval_grader = PromptTemplate(
    template = """
    RETURN ONLY JSON OBJECT WITH A SINGLE KEY 'score' AND NO ADDITIONAL TEXT.
    You are evaluating whether a retrieved document is relevant to a user question.
    If the document contains keywords or concepts related to the question, consider it relevant. 
    Example output: {{"score": "yes"}} or {{"score": "no"}}
    
    Retrieved document:
    {document}

    User question:
    {question}
    """,
    input_variables=["question", "document"],
) | llm | JsonOutputParser()

header_selector = PromptTemplate(
    template="""
    You are an assistant helping to identify which document sections are relevant to a user's medical question.
    Given the question and the list of document headers, select only the headers that would be useful to answer the question.
    Return a JSON object with a single key 'relevant_headers' and the value as a list of selected headers. 
    Do not include any additional text.

    Question:
    {question}

    Headers:
    {headers}
    """,
    input_variables=["question", "headers"],
) | llm | JsonOutputParser()


rag_chain = PromptTemplate(
    template="""
    Always answer in Polish.
    You are an assistant specialized in answering questions about medicinal products.
    Use the provided context to answer the question. If the answer is not found in the context, respond with "Niestety nie wiem."
    Keep your response quick and concise.

    Question: {question}  
    Context: {context}  
    Answer:
    """,
    input_variables=["question", "document"],
) | llm | StrOutputParser()

hallucination_grader = PromptTemplate(
    template="""
    You are a grader evaluating whether an answer is grounded in a given set of facts.
    Determine if the answer is supported by the provided information. 
    Return a binary score: 'yes' if the answer is grounded in the facts, 'no' if it is not.
    Output only a JSON object with a single key 'score' with a 'yes' or 'no' value and no additional text.
    
    Here are the facts:
    {documents} 

    Here is the answer: 
    {generation}
    """,
    input_variables=["generation", "documents"],
) | llm | JsonOutputParser()

answer_grader = PromptTemplate(
    template="""
    You are a grader evaluating whether an answer is useful in resolving a given question.
    Return a binary score: 'yes' if the answer is helpful in addressing the question, 'no' if it is not.
    Output only a JSON object with a single key 'score' and no additional text.
     
    Here is the answer:
    {generation} 

    Here is the question: {question}
    """,
    input_variables=["generation", "question"],
) | llm | JsonOutputParser()

question_router = PromptTemplate(
    template="""
    You are an expert at routing user questions to the appropriate data source: vectorstore or web search.
    Use the vectorstore for questions about medicinal products, even if keywords are only loosely related.
    For all other questions, choose web search.
    Return a binary choice: 'vectorstore' or 'web_search'.
    Output only a JSON object with a single key 'datasource' and no additional text.
    
    Question to route: 
    {question}
    """,
    input_variables=["question"],
) | llm | JsonOutputParser()

### --- NODE FUNCTIONS ---
def retrieve(state):
    """
    Retrieve documents from vectorstore

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, documents, that contains retrieved documents
    """
    print("---RETRIEVE---")
    question = state["question"]
    relevant_headers = state.get("relevant_headers", [])
    print(state.get("relevant_headers"))
    docs = load_texts_from_ids(extract_ids(query_hybrid_search(question, limit=3)))
    doc_txt = docs[next(iter(docs))]
    filtered_docs = headers_extraction(doc_txt, relevant_headers)
    if not filtered_docs:
        print("---WARNING: No sections matched headers. Using fallback excerpt.---")
        fallback = '\n'.join(doc_txt.split('\n')[:len(doc_txt.split('\n'))//5])
        documents = [fallback]
    else:
        documents = ['\n'.join(filtered_docs)]
    print(f"Filtered documents: {documents}")

    return {"documents": documents, "question": question}

def generate(state):
    """
    Generate answer using RAG on retrieved documents

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): New key added to state, generation, that contains LLM generation
    """
    print("---GENERATE---")
    question = state["question"]
    documents = state["documents"]
    
    generation = rag_chain.invoke({"context": documents, "question": question})
    return {"documents": documents, "question": question, "generation": generation}

def select_headers(state):
    """
    Determines which headers are relevant to the user's question.
    
    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Adds 'relevant_headers' to state
    """
    print("---SELECT HEADERS RELEVANT TO QUESTION---")
    question = state["question"]
    selected = header_selector.invoke({"question": question, "headers": headers})
    relevant_headers = selected['relevant_headers']
    print(f"Selected headers: {relevant_headers}")
    return {"question": question, "relevant_headers": relevant_headers}

def grade_documents(state):
    """
    Determines whether the retrieved documents are relevant to the question
    If any document is not relevant, we will set a flag to run web search

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Filtered out irrelevant documents and updated web_search state
    """

    print("---CHECK DOCUMENT RELEVANCE TO QUESTION---")
    question = state["question"]
    documents = state["documents"]
    filtered_docs = []
    web_search = "No"

    for d in documents:
        score = retrieval_grader.invoke({"question": question, "document": d})
        grade = score['score']

        if grade.lower() == "yes":
            print("---GRADE: DOCUMENT RELEVANT---")
            filtered_docs.append(d)
        else:
            print("---GRADE: DOCUMENT NOT RELEVANT---")
            web_search = "Yes"
            continue
    return {"documents": filtered_docs, "question": question, "web_search": web_search}

def web_search(state):
    """
    Web search based on the question

    Args:
        state (dict): The current graph state

    Returns:
        state (dict): Appended web results to documents
    """

    print("---WEB SEARCH---")
    question = state["question"]
    documents = state.get("documents", [])
    return {"documents": '', "question": question}
    docs = web_search_tool.invoke({"query": question})
    web_results = "\n".join(docs)
    web_results = Document(page_content=web_results)
    if documents is not None:
        documents.append(web_results)
    else:
        documents = [web_results]
    documents = documents[:3] 
    return {"documents": documents, "question": question}

def route_question(state):
    """
    Route question to web search or RAG.

    Args:
        state (dict): The current graph state

    Returns:
        str: Next node to call
    """

    print("---ROUTE QUESTION---")
    question = state["question"]
    print(question)
    source = question_router.invoke({"question": question})  
    print(source)

    if source['datasource'] == 'web_search':
        print("---ROUTE QUESTION TO WEB SEARCH---")
        return "websearch"
    elif source['datasource'] == 'vectorstore':
        print("---ROUTE QUESTION TO RAG---")
        return "vectorstore"
    else:
        print("---ROUTE DEFAULT FALLBACK TO VECTORSTORE---")
        return "vectorstore"
    
def decide_to_generate(state):
    """
    Determines whether to generate an answer, or add web search

    Args:
        state (dict): The current graph state

    Returns:
        str: Binary decision for next node to call
    """

    print("---ASSESS GRADED DOCUMENTS---")
    question = state["question"]
    web_search = state["web_search"]
    filtered_documents = state["documents"]

    if web_search == "Yes":
        print("---DECISION: ALL DOCUMENTS ARE NOT RELEVANT TO QUESTION, INCLUDE WEB SEARCH---")
        return "websearch"
    else:
        print("---DECISION: GENERATE---")
        return "generate"
    

def grade_generation_v_documents_and_question(state):
    """
    Determines whether the generation is grounded in the document and answers question.

    Args:
        state (dict): The current graph state

    Returns:
        str: Decision for next node to call
    """

    print("---CHECK HALLUCINATIONS---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generation"]
    
    score = hallucination_grader.invoke({"documents": documents, "generation": generation})
    print(f"Score: {score}")
    
    try:
        grade = score["score"]
    except (KeyError, TypeError, ValueError) as e:
        print("Error parsing score:", score)
    if grade != "yes" and grade != "no":
        print("Error parsing score:", score)

    if grade == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS---")
        print("---GRADE GENERATION vs QUESTION---")
        score = answer_grader.invoke({"question": question,"generation": generation})
        grade = score['score']
        
        if grade == "yes":
            print("---DECISION: GENERATION ADDRESSES QUESTION---")
            return "useful"
        else:
            print("---DECISION: GENERATION DOES NOT ADDRESS QUESTION---")
            return "not useful"
    else:
        print("---DECISION: GENERATION IS NOT GROUNDED IN DOCUMENTS, RE-TRY---")
        return "not supported"


### --- BUILDER FUNCTION ---
def build_graph():
    workflow = StateGraph(GraphState)
    
    workflow.add_node("select_headers", select_headers)
    workflow.add_node("websearch", web_search)
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("generate", generate)

    workflow.set_conditional_entry_point(
        route_question,
        {
            "websearch": "websearch",
            "vectorstore": "select_headers",
        },
    )

    workflow.add_edge("select_headers", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {
            "websearch": "websearch",
            "generate": "generate",
        },
    )
    workflow.add_edge("websearch", "generate")
    workflow.add_conditional_edges(
        "generate",
        grade_generation_v_documents_and_question,
        {
            "not supported": "generate",
            "useful": END,
            "not useful": "websearch",
        },
    )

    return workflow.compile()
