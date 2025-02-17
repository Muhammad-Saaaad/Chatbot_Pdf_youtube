import io

import PyPDF2
from fastapi import FastAPI, status, File, UploadFile
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

from backend.schema import GetYtLinks
from backend.rag_func import insert_pdf, insert_links , retrieved_data

app = FastAPI()

llm = ChatGroq(model="llama3-70b-8192", #model="llama3-8b-8192",
               max_tokens=700,
               temperature=0.1,
               max_retries=2,
               model_kwargs={"top_p":0.6,"presence_penalty" : 0.8}
)

@app.post("/Upload_links", status_code=status.HTTP_201_CREATED)
async def upload_link(req : GetYtLinks):
    yt_links = req.links
    insert_links(yt_links)

    return {"content":"links inserted sucessfully"}

@app.post("/Upload_pdf", status_code=status.HTTP_201_CREATED)
async def upload_file(file :UploadFile = File(...)):

    content = await file.read()
    pdf_file = io.BytesIO(content)
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()

    insert_pdf(text) # insert real links here to create a vector db

    return {"content":"pdf inserted sucessfully"}  

@app.get("/retrieve_data", status_code=status.HTTP_200_OK)
async def chat_vector_store(query : str):

    docs = retrieved_data(query)

    system_message="""
                You will receive a document or multiple documents based on user input. 
                Summarize the content in a single paragraph or, if necessary, in 2-3 
                concise paragraphs that directly address the user's query without any 
                introductory phrases (e.g., do not start with 'Here is a summary'). 
                Ensure no important details are lost. If you do not receive any documents, 
                respond conversationally and answer casual questions, but if the query is unrelated
                to the provided documents, respond with: 'I do not know the answer to this
                """

    message = ChatPromptTemplate.from_messages(
        [
            ("system",system_message),
            ("human","Here is the query{query}, and here are the documents {docs}")
        ]
    )
    chain = message | llm | StrOutputParser()
    return {"result": chain.invoke({"query":query,"docs":docs})}