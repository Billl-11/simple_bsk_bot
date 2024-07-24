from langchain_community.llms import Ollama
from langchain_community.document_loaders import WebBaseLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnablePassthrough
import streamlit as st
import warnings
import os

warnings.filterwarnings("ignore")

llm = Ollama(model="phi3", temperature=0.73)
embedding = OllamaEmbeddings(
        model="nomic-embed-text"
    )

path = "./Vectorstore_folder_2"

if os.path.exists(path):
    print(f'{path} found')
    vectorstore = Chroma(
        persist_directory = path,
        embedding_function = embedding
        )

else:

    #sites
    print(f'{path} not found')
    loader = WebBaseLoader(
        web_path=["https://barata.ai/about/",'https://barata.ai/','https://barata.ai/services/','https://barata.ai/products/','https://barata.ai/contact/']
    )

    docs = loader.load()

    #md
    markdown_path = "employees.md"
    loader = UnstructuredMarkdownLoader(markdown_path)
    additional_docs = loader.load()

    all_docs = docs + additional_docs

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 200,
        add_start_index = True 
    )
    all_splits = text_splitter.split_documents(all_docs)
    vectorstore = Chroma.from_documents(
        all_splits,
        embedding,
        persist_directory=path
        )
    print('Vector store creation is finish')


retriever = vectorstore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":2}
)

contextualize_q_system_prompt = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is. """

contextualize_q_prompt= ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()

qa_system_prompt = """You are an assistant for question-answering tasks all about Barata Sentosa Kencana (BSK) Company. \
Your name is BSK Bot. \
Use the following pieces of retrieved context to answer the question. \
If you don't know the answer, just say that you don't know. \
Use three sentences maximum and keep the answer concise.\
Keep the answer short.\

{context} """

qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def contextualized_question(input: dict):
    if input.get("chat_history"):
        return contextualize_q_chain
    else:
        return input["question"]

rag_chain = (
    RunnablePassthrough.assign(
        context = contextualized_question | retriever | format_docs
    )
    | qa_prompt 
    | llm
)

def response_generator(prompt, chat_history):
    ai_msg = rag_chain.invoke(
        {
            "question": prompt,
            "chat_history": chat_history
        }
    )
    return ai_msg


st.title("Simple BSK chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

   
    with st.chat_message("assistant"):
        response = response_generator(prompt, st.session_state.chat_history)
        st.markdown(response)
    
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.session_state.chat_history.extend(
        [HumanMessage(content=prompt), response]
    )