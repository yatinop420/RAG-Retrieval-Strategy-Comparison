
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq


# Initialize Groq LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


# Create Prompt Template
prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Answer the user's question using only the provided context.

If the answer is not available in the context,
clearly say that you don't know.

Context:
{context}

Question:
{question}

Answer:
""")


# Format Retrieved Documents
def format_docs(documents):
    return "\n\n".join(
        doc.page_content for doc in documents
    )


# Create RAG Generation Function
def generate_answer(question, documents):

    context = format_docs(documents)

    # Create LLM Chain
    chain = prompt | llm | StrOutputParser()

    # Generate Answer
    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response
