
   # pasar de un archivo json a una string para chattts

  #  archivo json tiene todas las lineas de dialogo que se tienen que transformar a voz pero solo DE UN PERSONAJE

 #   hay que organizar estas lineas de dialogo para que chattts pueda exportar los audios de forma organizada 
  #  u organizarlos despues

 #   pasar de json a un diccionario python

 #   con codigo dinamico 
# for loop para iteerar, enumerar las listas
# funcion filter caracter, para devolver una lista de todos los textos de un personaje
#guardar en un tuper con indice y texto


import json
import os

def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def extrac_dialogo(personaje):
    tuple_per = []
    for index, element in enumerate(json_dict["segments"]):
        if element["speaker"]==personaje:
            tuple_per.append((index,element["text"]))
    return tuple_per

def get_all_characters(json_data):
    #lista de personajes unitcos
    all_characters = set()
    
    # todos los personajes
    for segment in json_data["segments"]:
        if "speaker" in segment and segment["speaker"]:  # Check if speaker exists and is not empty
            all_characters.add(segment["speaker"])
    
    return all_characters


if __name__ == "__main__":
    file_path = r"C:\Users\JJota\Book2Speech\text.json"
    json_dict = load_json(file_path)
    
    # Get all unique characters
    all_characters = get_all_characters(json_dict)
    
    # Print the set of all characters
    print("All characters in the JSON file:")
    print(all_characters)