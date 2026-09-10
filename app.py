# fastapi -- python based web framework -- to build very fast api

from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import re
from fastapi.templating import Jinja2Templates # UI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

# initialize fastapi application
app = FastAPI(title="Text Summarizer App", description="Text Summarization using T5", version="1.0")

# model & tokenizer

model = T5ForConditionalGeneration.from_pretrained("./saved_summary_model")
tokenizer = T5Tokenizer.from_pretrained("./saved_summary_model")

# device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

model.to(device)

# templating
templates = Jinja2Templates(directory=".")

# Input schema for dialogue => string
class DialogueInput(BaseModel):
    dialogue: str

def clean_data(text):
    text = re.sub(r"\r\n", " ", text) # linespaces
    text = re.sub(r"\s+", " ", text)# whitespaces removal
    text = re.sub(r"<.*?>", " ", text) # html tags
    text = text.strip().lower()
    return text  

def summarize_dialogue(dialogue: str)-> str:
    dialogue = clean_data(dialogue) # clean the dialogue

    # tokenize
    inputs = tokenizer(
        dialogue,
        padding = "max_length",
        max_length = 512,
        truncation = True,
        return_tensor = "pt" # pt means pytorch tensor 
    ).to(device)

    # generate the summary => it will generate the summary token ids
    targets = model.generate(
        # targets is the token ids for 
        input_ids = inputs["input_ids"],
        attention_mask = inputs["attention_mask"],
        max_length = 150,
        num_beams = 4, # it means that model will return 4 different seq of outputs and then it will compare those 4 o/p and gives the best as output 
        early_stopping = True
    )

    # output token ids converted into summary text => decoding
    summary = tokenizer.decode(targets[0], skip_special_tokens = True) # special tokens like EOS, Separator
    return summary


# API endpoints
@app.post("/summarize/")
async def create_item(dialogue_input: DialogueInput):
    summary = summarize_dialogue(dialogue_input.dialogue)
    return {"summary":summary}

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html",{"request":request})