# Gui for pdf to tts using speech T5

#### Description:
the goal is this project is  create ui for reading adamic paper using tts model
orianly i want to Retrieval-based-Voice-Conversion-WebUI https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
to maketts but i have found a way to decode the model so i just use mircosoft t5 model instend
but i stiall have trouble using speecht5 to find out what dataset code for what voice
and the speecht5 have limit 512 token so i chunk the speech to 512 part each
origaly i wan to add all the function below but the project already take longer then i like
so i cut most of the fuction that i dot know how to do that may take me over a week to figure out how to do
also i just find out you cant make unti test (pytest) on gui
i can tran to use auto clicker and some machine learning to do the auto testing but this will take another week so here is the rush finally result

nougat-ocr only work with gpu if you using cpu use pdpdf2


nougat-ocr only work with gpu if you using cpu use pdpdf2

and nouget give a cleaner text
https://github.com/facebookresearch/nougat

speech t5 from mircosoft

https://huggingface.co/microsoft/speecht5_tts

- [x] gui
- [x] prograss bar
- [x] regex
- [x] option to use nougt

whishfull goal

- [ ] clean ui
- [ ] fast gui load time
- [ ] use Hypothesis for unti test
- [ ] use rvc model for tts to make human like voice
- [ ] find a way to load model outside of cache libary
- [ ] ask to intall hugging face model on first time set up
- [ ] add differnt voice option from cmu-arctic-xvectors
