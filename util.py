import json
import os
from PyPDF2 import PdfReader

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def loopeo(personaje, json_dict):
    tuple_per = []
    for index, element in enumerate(json_dict["segments"]):
        if element["speaker"]==personaje:
            tuple_per.append((index,element["text"]))
    return tuple_per

def load_pdf_test(path):

    try:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            text += page.extract_text()
            
        return text
    except Exception as e:
        print(f"Error reading PDF file: {str(e)}")
        return None

