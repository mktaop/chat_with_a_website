#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 15:27:56 2024

@author: avi_patel
"""

import streamlit as st
import bs4, os
from langchain import hub
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage

def setup_page():
    st.set_page_config(page_title="⚡ Chat with a website")
    header_text1 = "Chat with a website"
    st.header(f"      :blue[{header_text1}]", anchor=False)
    st.sidebar.title("Options")
    #st.title("Chat with websites")
    
    
def get_vectorstore_from_url(url):
    
    loader = WebBaseLoader(url)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter()
    splits = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings())
    return vectorstore


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_contexualize_q_prompt(llm):
    
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    contextualize_q_chain = contextualize_q_prompt | llm | StrOutputParser()
    return contextualize_q_chain


def get_qa_prompt():
    
    qa_system_prompt = """You are an assistant for question-answering tasks. \
    Use the following pieces of retrieved context to answer the question. \
    If you don't know the answer, just say that you don't know. \
    Use three sentences maximum and keep the answer concise.\
    
    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}"),
        ]
    )
    return qa_prompt



def get_clear():
    clear_button=st.sidebar.button("Clear Conversation", key="clear")
    return clear_button


def main():
    setup_page()
    clear = get_clear()
    
    website_url = st.text_input("Provide the website address and hit enter.")
    
    if website_url:
        vectorstore = get_vectorstore_from_url(website_url)
        retriever = vectorstore.as_retriever()
        
        message = []
        message.append({"role": "assistant", "content": "How may I help you?"})
        with st.chat_message(message[0]["role"],avatar="🧞‍♀️"):
            st.write(message[0]["content"])

        #prompt = hub.pull("rlm/rag-prompt")
        llm = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)
        
        contextualize_q_chain = get_contexualize_q_prompt(llm)
        qa_prompt = get_qa_prompt()
        
        def contextualized_question(input: dict):
            if input.get("chat_history"):
                return contextualize_q_chain
            else:
                return input["question"]
        
        rag_chain = (
            RunnablePassthrough.assign(
                context=contextualized_question | retriever | format_docs
            )
            | qa_prompt
            | llm
        )
        
        if clear not in st.session_state:
            chat_history = []
            
            question = st.chat_input("Type your question here and hit return.")
            if question:
                with st.chat_message("assistant", avatar="🧞‍♀️"):
                    with st.spinner("Thinking..."):
                        ai_msg = rag_chain.invoke({"question": question, "chat_history": chat_history})
                        chat_history.extend([HumanMessage(content=question), ai_msg])
                        response= ai_msg.content
                        st.write(response)
        
                        #second_question = "What are common ways of doing it?"
                        #rag_chain.invoke({"question": second_question, "chat_history": chat_history})
        
 
                               
if __name__ == '__main__':
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    main()



