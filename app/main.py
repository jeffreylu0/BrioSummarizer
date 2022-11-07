from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from spacy.lang.en import English
from onnx_inference import BrioOnnxPipeline
from enum import Enum
from pathlib import Path
from utils.download_models import download
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

start = time.time()
print("Loading BRIO pipelines...")
pegasus_summarizer = BrioOnnxPipeline(pegasus_checkpoint, list(map(str,pegasus_model_paths)))
bart_summarizer = BrioOnnxPipeline(bart_checkpoint, list(map(str,bart_model_paths)), pegasus=False)
end = time.time()
print(f"BRIO pipelines loaded in {end-start:.3f} seconds")

nlp = English()
spacy_tokenizer = nlp.tokenizer

class DocumentRequest(BaseModel):
    text: str = Field(min_length=100, title='Input Document')

class SummaryResponse(BaseModel):
    text: str
    time: float

class ModelName(str, Enum):
    pegasus = 'pegasus'
    bart = 'bart'

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
    if num_tokens < 200:
        summary = pegasus_summarizer(request.text)
        time_elapsed = time.time() - start
        return SummaryResponse(text=summary,time=time_elapsed)
    else:
        summary = bart_summarizer(request.text)
        time_elapsed = time.time() - start
        return SummaryResponse(text=summary, time=time_elapsed)

@app.post('/predict/{model_name}', response_model=SummaryResponse)
async def predict_with_model(request: DocumentRequest, model_name: ModelName):
    if model_name is ModelName.pegasus:
        return SummaryResponse(text=pegasus_summarizer(request.text))
    else:
        return SummaryResponse(text=bart_summarizer(request.text))

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    uvicorn.run(app, host='0.0.0.0', port=8000)
    

