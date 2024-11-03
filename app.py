from llama_index.core import SimpleDirectoryReader, SummaryIndex

import gradio as gr
import os
import shutil

from llama_index.core import Settings
from models import Qaya


index = None

def ingest(folder= './data'):
    # Load the your data
    documents = SimpleDirectoryReader(folder).load_data()
    global index
    index = SummaryIndex.from_documents(documents)


def rag(query):
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    return response.response


# define our LLM
Settings.llm = Qaya()

# define embed model
Settings.embed_model = "local:intfloat/multilingual-e5-large"

def upload_file(files):
    PATH = './data'
    if not os.path.exists(PATH):
        os.mkdir(PATH)
    for file in files:
        name = file.name.split('/')[-1]
        shutil.copy(file.name, PATH + '/' + name)
    ingest(PATH)

with gr.Blocks() as demo:
    gr.Markdown('## LLM Demo')

    file_output = gr.File()
    upload_button = gr.UploadButton("Click to Upload a File", file_types=['file'], file_count="multiple")
    upload_button.upload(upload_file, upload_button, file_output)

    with gr.Tab('Text'):
        txt_input = gr.Textbox(label = 'Query', type = 'text', placeholder = 'Enter your query here', lines = 2)
        txt_button = gr.Button('Answer')
        txt_output = gr.Textbox(label = 'Answer', type = 'text', placeholder = 'the answere',lines = 2)

    txt_button.click(rag, inputs=[txt_input], outputs=txt_output)

demo.queue().launch(share=True, debug=True)