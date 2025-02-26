
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

def loopeo(personaje):
    tuple_per = []
    for index, element in enumerate(json_dict["segments"]):
        if element["speaker"]==personaje:
            tuple_per.append((index,element["text"]))
    return tuple_per

if __name__ == "__main__":
    file_path = r"C:\Users\JJota\Downloads\chattts\ChatTTS\text.json"
    json_dict = load_json(file_path)
    personaje="Davos"
    personaje_dict=loopeo(personaje)
    if json_dict is not None:
        print("Extracted Dictionary:")
        print(personaje_dict)
