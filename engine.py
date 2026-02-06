from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def get_rag_chain(retriever):
    template = """You are an IPL History & Analytics Expert.
    Context: {context}
    Question: {question}
    
    Expert Answer:"""
    
    prompt = ChatPromptTemplate.from_template(template)
    model = ChatOpenAI(model="gpt-4o", temperature=0)

    # Modern LCEL Chain
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    return chain