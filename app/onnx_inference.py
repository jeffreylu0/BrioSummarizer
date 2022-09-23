from onnxruntime import InferenceSession, SessionOptions, ExecutionMode, GraphOptimizationLevel
import torch
from transformers import AutoConfig, PegasusTokenizer, BartTokenizer, PegasusForConditionalGeneration, BartForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from typing import Iterable
import functools
import operator

def create_onnx_sessions(model_paths: Iterable[str] = None, provider = ['CPUExecutionProvider']):

    path_to_encoder, path_to_decoder, path_to_init_decoder = model_paths

    options = SessionOptions()
    options.execution_mode = ExecutionMode.ORT_PARALLEL
    options.graph_optimization_level = GraphOptimizationLevel.ORT_ENABLE_ALL

    encoder_session = InferenceSession(path_to_encoder, options=options, provider=provider)
    decoder_session = InferenceSession(path_to_decoder, options=options, provider=provider)
    init_decoder_session = InferenceSession(path_to_init_decoder, options=options, provider=provider)

    return encoder_session, decoder_session, init_decoder_session

class BrioEncoder(torch.nn.Module):
    """
    Encoder utilizing InferenceSession
    """

    def __init__(self, encoder_session: InferenceSession = None):
        super().__init__()
        self.encoder = encoder_session

    def forward(
        self,
        input_ids,
        attention_mask,
        inputs_embeds=None,
        head_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Convert to numpy array for CPU inference, convert back to tensor
        encoder_hidden_state = torch.from_numpy(
            self.encoder.run(
                output_names = None,
                input_feed = {
                    'input_ids': input_ids.cpu().numpy(),
                    'attention_mask': attention_mask.cpu().numpy()
                }
            )[0]
        )

        return BaseModelOutput(encoder_hidden_state)

class BrioInitDecoder(torch.nn.Module):
    """
    Initial decoder utilizing InferenceSession
    """

    def __init__(self, init_decoder_session: InferenceSession = None):
        super().__init__()
        self.decoder = init_decoder_session

    def forward(
        self,
        input_ids,
        attention_mask,
        encoder_hidden_states
    ):
        decoder_outputs = self.decoder.run(
            output_names = None,
            input_feed = {
                'input_ids': input_ids.cpu().numpy(),
                'encoder_attention_mask': attention_mask.cpu().numpy(),
                'encoder_hidden_states': encoder_hidden_states.cpu().numpy()
            }
        )

        logits = torch.from_numpy(decoder_outputs[0])

        # convert flattented past_key_values to tuples that the transformer decoder expects
        flat_past_key_values = tuple(
            torch.from_numpy(x) for x in decoder_outputs[1:]
            )
        past_key_values = tuple(
            flat_past_key_values[i : i + 4] for i in range(0, len(flat_past_key_values), 4)
        )

        return logits, past_key_values

class BrioDecoder(torch.nn.Module):
    """
    Decoder utilizing InferenceSession
    """

    def __init__(self, decoder_session: InferenceSession = None):
        super().__init__()
        self.decoder = decoder_session

    def forward(
        self,
        input_ids,
        attention_mask,
        encoder_output,
        past_key_values: tuple
    ):

        flat_past_key_values = functools.reduce(operator.iconcat, past_key_values, [])
        past_key_values = {
            f"pkv_{i}": pkv.cpu().numpy() for i, pkv in enumerate(flat_past_key_values)
        }

        # Run decoder with previously calculated past_key_values from initial decoder        
        decoder_outputs = self.decoder.run(
            output_names = None, 
            input_feed = {
                'input_ids': input_ids.cpu().numpy(),
                'encoder_attention_mask': attention_mask.cpu().numpy(),
                **past_key_values}
                )
        logits = torch.from_numpy(decoder_outputs[0])
        # converts each value of the list to tensor from numpy
        list_pkv = tuple(torch.from_numpy(x) for x in decoder_outputs[1:])

        # creates a tuple of tuples of shape 6x4 from the above tuple
        output_past_key_values = tuple(
            list_pkv[i : i + 4] for i in range(0, len(list_pkv), 4)
        )

        return logits, output_past_key_values


class BrioPegasusOnnx(PegasusForConditionalGeneration):

    def __init__(self, model_checkpoint, onnx_sessions:Iterable[InferenceSession] = None):
        self.config = AutoConfig.from_pretrained(model_checkpoint)
        super().__init__(self.config)

        assert len(onnx_sessions) == 3, "Encoder, decoder, and initial decoder sessions required"
        encoder_session, decoder_session, init_decoder_session = onnx_sessions

        self.encoder = BrioEncoder(encoder_session)
        self.decoder = BrioDecoder(decoder_session)
        self.init_decoder = BrioInitDecoder(init_decoder_session)
    
    def get_encoder(self):
        
        return self.encoder
    
    def get_decoder(self):

        return self.decoder
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )

        encoder_hidden_states = encoder_outputs[0]

        if past_key_values is not None:
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        if past_key_values is None:

            # runs only for the first time:
            init_onnx_outputs = self.init_decoder(
                decoder_input_ids, attention_mask, encoder_hidden_states
            )

            logits, past_key_values = init_onnx_outputs

        else:

            onnx_outputs = self.decoder(
                decoder_input_ids,
                attention_mask,
                encoder_hidden_states,
                past_key_values,
            )

            logits, past_key_values = onnx_outputs

        return Seq2SeqLMOutput(logits=logits, past_key_values=past_key_values)

class BrioBartOnnx(BartForConditionalGeneration):

    def __init__(self, model_checkpoint, onnx_sessions:Iterable[InferenceSession] = None):
        self.config = AutoConfig.from_pretrained(model_checkpoint)
        super().__init__(self.config)

        assert len(onnx_sessions) == 3, "Encoder, decoder, and initial decoder sessions required"
        encoder_session, decoder_session, init_decoder_session = onnx_sessions

        self.encoder = BrioEncoder(encoder_session)
        self.decoder = BrioDecoder(decoder_session)
        self.init_decoder = BrioInitDecoder(init_decoder_session)
    
    def get_encoder(self):
        
        return self.encoder
    
    def get_decoder(self):

        return self.decoder
    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        if encoder_outputs is None:
            # Convert encoder inputs in embeddings if needed
            encoder_outputs = self.encoder(
                input_ids=input_ids, attention_mask=attention_mask
            )

        encoder_hidden_states = encoder_outputs[0]

        if past_key_values is not None:
            if decoder_input_ids is not None:
                decoder_input_ids = decoder_input_ids[:, -1:]
            if decoder_inputs_embeds is not None:
                decoder_inputs_embeds = decoder_inputs_embeds[:, -1:]

        if past_key_values is None:

            # runs only for the first time:
            init_onnx_outputs = self.init_decoder(
                decoder_input_ids, attention_mask, encoder_hidden_states
            )

            logits, past_key_values = init_onnx_outputs

        else:

            onnx_outputs = self.decoder(
                decoder_input_ids,
                attention_mask,
                encoder_hidden_states,
                past_key_values,
            )

            logits, past_key_values = onnx_outputs

        return Seq2SeqLMOutput(logits=logits, past_key_values=past_key_values)

class BrioOnnxPipeline:

    def __init__(self, 
                model_checkpoint: str = None, 
                model_paths: Iterable[str] = None, 
                pegasus = True,
                **generation_kwargs):
        
        onnx_sessions = create_onnx_sessions(model_paths)
        if pegasus:
            self.model = BrioPegasusOnnx(model_checkpoint, onnx_sessions)
            self.tokenizer = PegasusTokenizer.from_pretrained(model_checkpoint)
        else:
            self.model = BrioBartOnnx(model_checkpoint, onnx_sessions)
            self.tokenizer = BartTokenizer.from_pretrained(model_checkpoint)
        self.model.eval()
        self.gen_kwargs = generation_kwargs

    def __call__(self, document: str):
        
        encoded_input = self.tokenizer(document, max_length=500, truncation=True, return_tensors='pt') # input ids and attention mask
        output_token_ids = self.model.generate(**encoded_input, **self.gen_kwargs)
        output_text = self.tokenizer.batch_decode(output_token_ids, skip_special_tokens=True)[0]

        return output_text

def main():

    quantized_model_paths = ['./models/brio-xsum-cased-encoder-quantized.onnx',
                             './models/brio-xsum-cased-decoder-quantized.onnx', 
                             './models/brio-xsum-cased-init-decoder-quantized.onnx']

    model_checkpoint = 'Yale-LILY/brio-xsum-cased'

    text= """
        For a few years, rumors have persisted that Microsoft was exploring building some form 
        of streaming stick to offer Xbox Cloud Gaming via a more affordable dongle, similarly 
        to Chromecast and Google Stadia. The first hint was Project Hobart. More recently, a code 
        name "Keystone" appeared in an Xbox OS list, lending fire to rumors that Microsoft was 
        continuing to explore additional hardware for the Xbox lineup. 

        We can now confirm that that is indeed true, and it pertains to a modernized HDMI 
        streaming device that runs Xbox Game Pass and its cloud gaming service. Microsoft is, 
        however, taking exploring additional iterations of the product before taking it to market. 
        """

    pipeline = BrioOnnxPipeline(model_checkpoint, quantized_model_paths)
    output = pipeline(text)
    print(output)
        
if __name__ == '__main__':
    main()