from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

TEMPLATE = """You are an IPL History & Analytics Expert.
Context: {context}
Question: {question}

Expert Answer:"""


def get_chain(retriever, model="claude-3-sonnet-20240229", temperature=0, **kwargs):
    prompt = ChatPromptTemplate.from_template(TEMPLATE)
    llm = ChatAnthropic(model=model, temperature=temperature)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain
