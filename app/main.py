from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from spacy.lang.en import English
from onnx_inference import BrioOnnxPipeline
from enum import Enum
import uvicorn
import time
import os

bart_model_paths = ['./models/bart/brio-cnndm-uncased-encoder-quantized.onnx',
                        './models/bart/brio-cnndm-uncased-decoder-quantized.onnx', 
                        './models/bart/brio-cnndm-uncased-init-decoder-quantized.onnx']

pegasus_model_paths = ['./models/pegasus/brio-xsum-cased-encoder-quantized.onnx',
                        './models/pegasus/brio-xsum-cased-decoder-quantized.onnx', 
                        './models/pegasus/brio-xsum-cased-init-decoder-quantized.onnx']

pegasus_checkpoint = 'Yale-LILY/brio-xsum-cased' 
bart_checkpoint = 'Yale-LILY/brio-cnndm-uncased'

start = time.time()
print("Loading BRIO pipelines...")
pegasus_summarizer = BrioOnnxPipeline(pegasus_checkpoint, pegasus_model_paths)
bart_summarizer = BrioOnnxPipeline(bart_checkpoint, bart_model_paths, pegasus=False)
end = time.time()
print(f"BRIO pipelines loaded in {end-start:.3f} seconds")

nlp = English()
spacy_tokenizer = nlp.tokenizer

class DocumentRequest(BaseModel):
    text: str = Field(min_length=100, title='Input Document')

class SummaryResponse(BaseModel):
    text: str

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
    num_tokens = len(spacy_tokenizer(request.text))
    if num_tokens < 200:
        return SummaryResponse(text=pegasus_summarizer(request.text))
    else:
        return SummaryResponse(text=bart_summarizer(request.text))

@app.post('/predict/{model_name}', response_model=SummaryResponse)
async def predict_with_model(request: DocumentRequest, model_name: ModelName):
    if model_name is ModelName.pegasus:
        return SummaryResponse(text=pegasus_summarizer(request.text))
    else:
        return SummaryResponse(text=bart_summarizer(request.text))

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    uvicorn.run('main:app', host='0.0.0.0', port=8000, reload=True)

