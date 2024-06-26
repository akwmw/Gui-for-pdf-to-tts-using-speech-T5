import os
import re
import PyPDF2
import numpy as np
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from datasets import load_dataset
import torch
import soundfile as sf
from moviepy.editor import *
import tkinter as tk
from tkinter import Tk, Frame, Button, Label, StringVar, ttk, Radiobutton, IntVar, messagebox, filedialog
from pathlib import Path
from pylatexenc.latex2text import LatexNodes2Text
selected_tool = None
nocget_selected = False
def convert_pdf_to_txt(file_path):
    with open(file_path, 'rb') as f:
        reader = PyPDF2.PdfFileReader(f)
        num_pages = reader.numPages
        text = ""
        for page in range(num_pages):
            text += reader.getPage(page).extractText()

        return text

def convert_text_to_audio(text):
    inputs = processor(text=text, return_tensors="pt").to(device)
    speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)
    return speech.cpu().numpy()

def process_pdf():
    global nocget_selected
    file_path = filedialog.askopenfilename()
    if not file_path:
        return
    if not file_path.lower().endswith('.pdf'):
        messagebox.showwarning("Warning", "This is not a PDF file")
        return
    if selected_tool is None:
        label_print.set("Please select a tool")
        return
    else:
        button.config(state=tk.DISABLED)


    if selected_tool == "pypdf2":
            with open(file_path, 'rb') as pdf_file:
                read_pdf = PyPDF2.PdfFileReader(pdf_file)
                num_pages = read_pdf.numPages
                txt = ""
                for page in range(num_pages):
                    txt += read_pdf.getPage(page).extractText()
            txt = convert_pdf_to_txt(file_path)
    elif selected_tool == "nocget":
        #avoid Error: mkl-service + Intel(R) MKL: MKL_THREADING_LAYER=INTEL is incompatible with libgomp-a34b3233.so.1 library.
        os.environ['MKL_THREADING_LAYER'] = 'GNU'
        nocget_selected = True
        input_pdf = file_path
        outdir = os.path.abspath(__file__)
        output_dir = os.path.dirname(outdir)
        cmd = 'nougat "{}" -o "{}" -m 0.1.0-base'.format(input_pdf, output_dir)
        os.system(cmd)
        #print(output_dir)
        suffix = '.mmd'

        name = os.path.join(output_dir,Path(file_path).stem + suffix)
        #print(name)
        with open(name , 'r') as file:
            txt = file.read()

    txt= process_line(txt)
    txt = txt = LatexNodes2Text().latex_to_text(txt)
    #call debug function towrite edited text to txt file
    #write_to_txt(txt, Path(file_path).stem)
    if nocget_selected == True:
        #remove temp mmd file
        os.remove(name)
    chunks = [txt[i: i + 512] for i in range(0, len(txt), 512)]

    audios = []
    label_print.set(f"Processing chunk 0/{len(chunks)}")
    progressbar['maximum'] = len(chunks)
    for idx, chunk in enumerate(chunks):
        label_print.set(f"Processing chunk {idx+1}/{len(chunks)}")
        root.update_idletasks()
        audio = convert_text_to_audio(chunk)
        audios.append(audio)
        progressbar['value'] = idx + 1  # Update the progress bar
        root.update_idletasks()  # Update the progress bar display
    output_audio = np.concatenate(audios)
    output_file_name = os.path.splitext(file_path)[0] + ".wav"
    sf.write(output_file_name, output_audio, samplerate=16000)
    button.config(state=tk.NORMAL)
    nocget_selected = False
    label_print.set("Completed!wav file is in same input path")
def remove_before_abstract(text):
    # Find the index where 'Abstract' starts
    match = re.search('Abstract', text)

    # If 'Abstract' is not found, return the original text
    if match is None:
        return text

    # Otherwise, return the text starting from 'Abstract'
    return text[match.start():]

def remove_after_ack_or_ref(text):
    # Find the index where 'Acknowledgements' or 'References' starts
    match = re.search('Acknowledgements|References', text)

    # If neither 'Acknowledgements' nor 'References' is found, return the original text
    if match is None:
        return text

    # Otherwise, return the text up to 'Acknowledgements' or 'References'
    return text[:match.start()]
def remove_table(line: str) -> str:
    pattern = r"\\begin\{table\}(.*?)\\end\{table\}"
    line = re.sub(pattern, '', line, flags=re.DOTALL)
    return line

def remove_references(text):
    clean_text = re.sub(r'\[\d+(;\s\d+)*\]', '', text)
    return clean_text
def process_line(line: str) -> str:
    line = remove_before_abstract(line)
    line = remove_after_ack_or_ref(line)
    line = remove_references(line)
    line = remove_table(line)
    return line

def write_to_txt(text, filename):
    #debug print to text function
    with open(filename, 'w') as f:
        f.write(text)

def set_tool(tool):
    global selected_tool
    selected_tool = tool



def main():
    #def main for legel reason
    return
root = Tk()
root.geometry("500x300")
root.title("PDF to speech")
frame = Frame(root, padx=10, pady=10)
frame.pack()
is_pypdf2_selected = tk.IntVar()

button = Button(frame, text="Select PDF File", command=process_pdf)
button.pack()

tk.Radiobutton(frame, text="PyPDF2", variable=is_pypdf2_selected, value=1, command=lambda: set_tool("pypdf2")).pack()
tk.Radiobutton(frame, text="Nougat", variable=is_pypdf2_selected, value=2, command=lambda: set_tool("nocget")).pack()

label_print = StringVar()
label = Label(frame, textvariable=label_print)
label.pack()
progressbar = ttk.Progressbar(root, length=200)
progressbar.pack()
selected_tool = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts").to(device)
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan").to(device)

embeddings_dataset = load_dataset("Matthijs/cmu-arctic-xvectors", split="validation")
speaker_embeddings = torch.tensor(embeddings_dataset[7306]["xvector"]).unsqueeze(0).to(device)

root.mainloop()

