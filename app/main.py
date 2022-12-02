from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from spacy.lang.en import English
from transformers import (PegasusTokenizer, 
                          BartTokenizer, 
                          PegasusForConditionalGeneration, 
                          BartForConditionalGeneration)
from .onnx_inference import BrioOnnxPipeline
from enum import Enum
from pathlib import Path
from .utils.download_models import download
import uvicorn
import time
import os

# Use local model checkpoints or not
local = False

# Model paths
model_dir = Path.cwd() / "models"
pegasus_model_paths = [model_dir / 'brio-xsum-cased-encoder-quantized.onnx',
                       model_dir / 'brio-xsum-cased-decoder-quantized.onnx', 
                       model_dir / 'brio-xsum-cased-init-decoder-quantized.onnx']

bart_model_paths = [model_dir / 'brio-cnndm-uncased-encoder-quantized.onnx',
                    model_dir / 'brio-cnndm-uncased-decoder-quantized.onnx', 
                    model_dir / 'brio-cnndm-uncased-init-decoder-quantized.onnx']

model_paths = pegasus_model_paths + bart_model_paths

# If models are not in the directory, download them from S3
if not all([path.exists() for path in model_paths]):
    print('Downloading models...')
    download()
    print('Models downloaded')

# Local model checkpoints
if local:
    pegasus_checkpoint = './models/brio-xsum-cased' 
    bart_checkpoint = './models/brio-cnndm-uncased'
else:
    pegasus_checkpoint = 'Yale-LILY/brio-xsum-cased'
    bart_checkpoint = 'Yale-LILY/brio-cnndm-uncased'

# Load ONNX models
print("Loading BRIO ONNX pipelines...")
pegasus_summarizer = BrioOnnxPipeline(pegasus_checkpoint, list(map(str,pegasus_model_paths)))
bart_summarizer = BrioOnnxPipeline(bart_checkpoint, list(map(str,bart_model_paths)), pegasus=False)
print(f"BRIO ONNX pipelines loaded!")

# Load PyTorch models (for speed test mainly)
print("Loading BRIO PyTorch models and tokenizers...")
pegasus_model = PegasusForConditionalGeneration.from_pretrained(pegasus_checkpoint)
pegasus_tokenizer = PegasusTokenizer.from_pretrained(pegasus_checkpoint)
bart_model = BartForConditionalGeneration.from_pretrained(bart_checkpoint)
bart_tokenizer = BartTokenizer.from_pretrained(bart_checkpoint)
print("BRIO PyTorch models and tokenizers loaded!")

# Load spacy tokenizer to check number of tokens in request
nlp = English()
spacy_tokenizer = nlp.tokenizer

class DocumentRequest(BaseModel):
    text: str = Field(min_length=100, title='Input Document')

class SummaryResponse(BaseModel):
    text: str
    time: float
    model: str
    framework: str

# Enumerations for testing models and frameworks
class ModelName(str, Enum):
    pegasus = 'pegasus'
    bart = 'bart'
    
class Framework(str, Enum):
    pytorch = 'pytorch'
    onnx = 'onnx'

app = FastAPI()

# allow CORS requests from any host so that the JavaScript can communicate with the server
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

@app.get('/')
def index():
    return "BRIO Summarizer"

@app.post('/predict', response_model=SummaryResponse)
async def predict(request: DocumentRequest):
    start = time.time()
    num_tokens = len(spacy_tokenizer(request.text))
    # Given shorter source document, use abstractive summarization
    if num_tokens < 50:
        summary = pegasus_summarizer(request.text)
        time_elapsed = time.time() - start
        return SummaryResponse(text=summary,time=time_elapsed, model='pegasus', framework='onnx')

    # Given longer source document, use extractive summarization
    else:
        summary = bart_summarizer(request.text)
        time_elapsed = time.time() - start
        return SummaryResponse(text=summary, time=time_elapsed, model='bart', framework='onnx')

# Enumeration for testing a specific model
@app.post('/predict/{model_name}/{framework}', response_model=SummaryResponse)
async def predict_with_model(request: DocumentRequest, 
                             model_name: ModelName,
                             framework: Framework):

    if model_name is ModelName.pegasus:

        if framework is Framework.pytorch:

            start = time.time()
            encoded_input = pegasus_tokenizer(request.text,return_tensors='pt')
            output_tokens = pegasus_model.generate(**encoded_input)
            summary = pegasus_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
            time_elapsed = time.time() - start
            return SummaryResponse(text=summary, time=time_elapsed, model=model_name, framework=framework)

        if framework is Framework.onnx:
            start = time.time()
            summary = pegasus_summarizer(request.text)
            time_elapsed = time.time() - start
            return SummaryResponse(text=summary, time=time_elapsed, model=model_name, framework=framework)

    if model_name is ModelName.bart:

        if framework is Framework.pytorch:
            start = time.time()
            encoded_input = bart_tokenizer(request.text,return_tensors='pt')
            output_tokens = bart_model.generate(**encoded_input)
            summary = bart_tokenizer.batch_decode(output_tokens, skip_special_tokens=True)[0]
            time_elapsed = time.time() - start
            return SummaryResponse(text=summary, time=time_elapsed, model=model_name, framework=framework)
        
        if framework is Framework.onnx:
            start = time.time()
            summary = bart_summarizer(request.text)
            time_elapsed = time.time() - start
            return SummaryResponse(text=summary, time=time_elapsed, model=model_name, framework=framework)           

if __name__ == '__main__':
    #os.environ['KMP_DUPLICATE_LIB_OK']='True'
    uvicorn.run(app, host='0.0.0.0', port=80)
    

