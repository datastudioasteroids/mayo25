🇦🇷 Data Studio Asteroids — Efemérides 25 de Mayo
“En esta fecha tan significativa para nuestro país, conmemoramos el 25 de mayo de 1810, día en que se dio inicio al proceso que conduciría a nuestra independencia. Fue entonces cuando un grupo de ciudadanos valientes y comprometidos decidió asumir el destino común, abriendo paso a una nueva etapa de autonomía, diálogo y construcción colectiva.

Hoy, más de dos siglos después, nos reunimos con ese mismo espíritu para presentar Data Studio Asteroids, un espacio nacido con el propósito de democratizar el acceso al conocimiento, potenciar la innovación y fomentar la colaboración en el universo de los datos.

Así como en 1810 se soñó con una patria libre y soberana, nosotros creemos en una nueva soberanía: la del pensamiento crítico, la de la información abierta, y la de las herramientas tecnológicas al servicio del desarrollo humano.

Este lanzamiento no es solo un proyecto. Es una declaración de principios. Es una invitación a explorar, a crear, a decidir con responsabilidad y visión. Porque construir futuro también es un acto de libertad.

Que esta jornada patria nos inspire a seguir impulsando ideas que transformen, que unan y que abran caminos.

Muchas gracias.”

🚀 Descripción del Proyecto
Data Studio Asteroids - mayo25 es una experiencia interactiva impulsada por un modelo de lenguaje de última generación y una arquitectura RAG (Retrieval-Augmented Generation). Este proyecto conmemora el espíritu de emancipación del 25 de mayo a través de tecnología, historia y participación ciudadana.

🧠 ¿Qué incluye el sistema?
📌 LLM + RAG
Este proyecto combina un LLM (Large Language Model) con arquitectura RAG para responder preguntas en lenguaje natural sobre los sucesos de la Revolución de Mayo de 1810, utilizando:

Un backend Python con integración a un modelo LLM (como HuggingChat, GPT o LLaMA).

Sistema RAG que combina:

Recuperación de información histórica desde documentos (La Revolución de Mayo de 1810.docx).

Generación de respuestas contextualizadas.

Aprendizaje continuo a partir del diálogo en consola (chat_console.py).

🧩 Estructura del Proyecto
php
Copiar
Editar
EFEMERIDES_25M/
├── backend/
│   ├── chat_console.py         # Consola de chat LLM
│   ├── modelo.py               # Lógica de carga y RAG
│   ├── server.py               # Servidor backend
│   └── cookies.json            # Tokens y credenciales (si aplica)
│
├── frontend/
│   └── index.html              # Interfaz visual de la experiencia
│
├── mayo25/                     # Librerías de soporte para entorno virtual
│   ├── Include/
│   ├── Lib/
│   ├── Scripts/
│   └── share/
│
├── static/                     # Recursos gráficos y multimedia
│   ├── fondo.png
│   ├── guemes.png
│   ├── juana.png
│   ├── manuel_belgrano.png
│   ├── milicias_urbanas.png
│   ├── pueblo.jpeg
│   ├── script.js
│   └── styles.css
│
├── La Revolución de Mayo de 1810.docx  # Documento histórico base para RAG
├── requirements.txt                    # Dependencias del proyecto
🛠️ Requisitos
Instala las dependencias necesarias con:

bash
Copiar
Editar
pip install -r requirements.txt
Asegúrate de contar con acceso a un modelo LLM (puede ser local o remoto), y configurar las credenciales si aplica en cookies.json.
