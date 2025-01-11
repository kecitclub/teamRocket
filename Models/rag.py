import google.generativeai as genai
import os
import pdfplumber
import uvicorn
from fastapi import FastAPI, File, UploadFile, HTTPException

app = FastAPI()

genai.configure(api_key="AIzaSyAU9duCSva2f3XfdJKCEr80VeE3ZjxvW6g")
model = genai.GenerativeModel("gemini-1.5-flash")
# response = model.generate_content("expain about the highest mountain in the world")
# print(response.text)

def extract_text_from_pdf(pdf_file):
    text = ""
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
        return text
    
text = extract_text_from_pdf("The Clockmaker's Secret.pdf")
# print(text)
# print("-------------------------------------------------------------------------------")

def summarize_text(text):
    try:
        response = model.generate_content([f"please summarize the following text:\n\n{text}"])
        return response.text
    except Exception as e:
        return f"An error occured: {e}"

# summary = summarize_text(text)
# print(summary)
# print("-------------------------------------------------------------------------------")

def question_text(text, question):
    try:
        response = model.generate_content([f"Please answer the following question based on the text:\n\nText: {text}\n\nQuestion: {question}"])
        return response.text
    except Exception as e:
        return f"An error occured: {e}"

answer = question_text(text,'who are the author of the given paper?')
print(answer)
# print("-------------------------------------------------------------------------------")
def generate_questions(text):
    try:    
        response = model.generate_content([f"Please generate questions based on the text:\n\nText: {text}"])
        return response.text
    except Exception as e:
        return f"An error occured: {e}"

# question = question_text(text)
# print(question)

# @app.post("/upload-pdf/")
# async def upload_pdf(file: UploadFile = File(...)):
#     try:
#         with open(file.filename, "wb") as buffer:
#             buffer.write(await file.read())
#         text = extract_text_from_pdf(file.filename)
#         os.remove(file.filename)  # Cleanup uploaded file
#         return {"text": text}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

@app.post("/summarize/")
async def summarize(file: UploadFile = File(...)):
    try:
        with open(file.filename, "wb") as buffer:
            buffer.write(await file.read())
        text = extract_text_from_pdf(file.filename)
        os.remove(file.filename)  # Cleanup uploaded file
        summary = summarize_text(text)
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

@app.post("/question/")
async def ask_question(file: UploadFile = File(...), question: str = ""):
    try:
        with open(file.filename, "wb") as buffer:
            buffer.write(await file.read())
        text = extract_text_from_pdf(file.filename)
        os.remove(file.filename)  # Cleanup uploaded file
        answer = question_text(text, question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

@app.post("/generate-questions/")
async def generate(file: UploadFile = File(...)):
    try:
        with open(file.filename, "wb") as buffer:
            buffer.write(await file.read())
        text = extract_text_from_pdf(file.filename)
        os.remove(file.filename)  # Cleanup uploaded file
        questions = generate_questions(text)
        return {"questions": questions}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {e}")

# if __name__ == "__main__":
#     # uvicorn.run(app, host="0.0.0.0", port=8000)
#     uvicorn.run(app, port=8000)
