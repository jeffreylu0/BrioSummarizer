import requests
from pathlib import Path

s3_urls = ['https://brio-onnx-models.s3.amazonaws.com/bart/brio-cnndm-uncased-decoder-quantized.onnx',
           'https://brio-onnx-models.s3.amazonaws.com/bart/brio-cnndm-uncased-encoder-quantized.onnx',
           'https://brio-onnx-models.s3.amazonaws.com/bart/brio-cnndm-uncased-init-decoder-quantized.onnx',
           'https://brio-onnx-models.s3.amazonaws.com/pegasus/brio-xsum-cased-decoder-quantized.onnx', 
           'https://brio-onnx-models.s3.amazonaws.com/pegasus/brio-xsum-cased-encoder-quantized.onnx', 
           'https://brio-onnx-models.s3.amazonaws.com/pegasus/brio-xsum-cased-init-decoder-quantized.onnx']

def download():
    for url in s3_urls:
        file_name = url.split('/')[-1]
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            Path('./models').mkdir(parents=True, exist_ok=True)
            with open(f'models/{file_name}', 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192): 
                    f.write(chunk)

    