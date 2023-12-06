#===================================================#
#===== Universidad Nacional Autónoma de México =====# 
#=====  Asistente virtual básico tipo Watson   =====#
#=====      Johan Fernando Romo Eligio         =====#
#===================================================#

import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# Cargar el modelo de spaCy para PLN
nlp = spacy.load("en_core_web_sm")

# Datos de entrenamiento (ejemplo simple)
training_data = [
    ("Watson es un asistente virtual avanzado.", "Watson"),
    ("Python es un lenguaje de programación poderoso.", "Programación"),
    ("Los chatbots están transformando la interacción.", "Chatbots"),
    # Agrega más datos etiquetados según sea necesario
]

# Separar datos en texto y etiquetas
texts, labels = zip(*training_data)

# Dividir datos en conjunto de entrenamiento y prueba
texts_train, texts_test, labels_train, labels_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Preprocesamiento de texto con spaCy
def preprocess_text(text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

# Crear un modelo de PLN y AA simple
model = make_pipeline(TfidfVectorizer(preprocessor=preprocess_text), MultinomialNB())

# Entrenar el modelo
model.fit(texts_train, labels_train)

# Evaluación del modelo
predictions = model.predict(texts_test)
accuracy = accuracy_score(labels_test, predictions)
print(f"Exactitud del modelo: {accuracy * 100:.2f}%")

# Interfaz de usuario simple
while True:
    user_input = input("Ingresa una consulta (o 'salir' para terminar): ")
    if user_input.lower() == 'salir':
        break
    prediction = model.predict([user_input])[0]
    print(f"Predicción: {prediction}")
