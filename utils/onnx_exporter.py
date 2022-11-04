from transformers import (PegasusConfig, 
                          PegasusForConditionalGeneration,
                          BartConfig,
                          BartForConditionalGeneration)
from export_model_structure import BrioEncoder, BrioDecoderLMHead, BrioDecoderLMHeadInitial
from onnxruntime.quantization import quantize_dynamic, QuantType
from typing import Union, Iterable
import torch
from pathlib import Path
from progress.bar import Bar
import functools
import operator
import os

def turn_model_into_encoder_decoder(model: Union[PegasusForConditionalGeneration, BartForConditionalGeneration] = None):
    encoder = model.get_encoder()
    decoder = model.get_decoder()
    lm_head = model.get_output_embeddings()
    final_logits_bias = model.final_logits_bias

    decoder_with_lm_head = BrioDecoderLMHead(decoder, lm_head, final_logits_bias, model.config)
    simplified_encoder = BrioEncoder(encoder)
    decoder_with_lm_head_init = BrioDecoderLMHeadInitial(decoder, lm_head, final_logits_bias, model.config)

    return simplified_encoder, decoder_with_lm_head, decoder_with_lm_head_init

def export_to_onnx(
    model_checkpoint: str = None,
    output_path: str = None,
    input_sequence_length=256,
    onnx_opset_version=12,
    pegasus=True
):

    if pegasus:
        print("Loading model checkpoint...")
        model = PegasusForConditionalGeneration.from_pretrained(model_checkpoint)
        model_config = PegasusConfig.from_pretrained(model_checkpoint)
        print("Model checkpoint loaded")

    else:
        print("Loading model checkpoint...")
        model = BartForConditionalGeneration.from_pretrained(model_checkpoint)
        model_config = BartConfig.from_pretrained(model_checkpoint)
        print("Model checkpoint loaded")

    simplified_encoder, decoder_with_lm_head, decoder_with_lm_head_init = turn_model_into_encoder_decoder(model)
    output_path = Path(output_path)
    encoder_path, decoder_path, init_decoder_path = get_model_paths(
        model_checkpoint, output_path, quantized=False
    )

    #Dummy inputs for encoder and decoder to trace model graph
    batch_size = 1 # not configurable since only CPU
    encoder_seq_length = input_sequence_length
    encoder_input_ids = torch.ones(batch_size, encoder_seq_length, dtype=torch.int64)
    encoder_attention_mask = torch.ones(batch_size, encoder_seq_length, dtype=torch.int64)
    encoder_output = torch.ones(
        (batch_size, encoder_seq_length, model_config.d_model), dtype=torch.float32)

    decoder_seq_length = 1 # a decoder sequence length is always one because it's just the last generated token
    decoder_input_ids = torch.ones(batch_size, decoder_seq_length, dtype=torch.int64)
    decoder_attention_mask = torch.ones(batch_size, encoder_seq_length, dtype=torch.int64)

    num_heads = model_config.decoder_attention_heads 
    d_kv = model_config.d_model // num_heads # embedding size per head

    #Self-attention size 1, 8, 1, 64
    sa = torch.ones(
        (batch_size, num_heads, decoder_seq_length, d_kv), dtype=torch.float32
    )  

    #Cross-attention size 1, 8, 1, 64
    ca = torch.ones(
        (batch_size, num_heads, encoder_seq_length, d_kv), dtype=torch.float32
    ) 

    bart_block = (sa, sa, ca, ca)
    past_key_values = (bart_block,) * model_config.decoder_layers

    flat_past_key_values = functools.reduce(operator.iconcat, past_key_values, [])

    decoder_all_inputs = tuple(
        [decoder_input_ids, decoder_attention_mask, encoder_output] + flat_past_key_values
    )

    # for progress bars
    bar = Bar("Exporting to onnx...", max=3)

    import warnings

    # ignores all the warnings during conversion
    warnings.filterwarnings("ignore")

    with torch.no_grad():

        decoder_inputs = [
            "input_ids",
            "encoder_attention_mask",
            "encoder_hidden_states",
        ]

        pkv_input_names = ["pkv_{}".format(i) for i in range(len(flat_past_key_values))] #past key values

        decoder_input_names = decoder_inputs + pkv_input_names

        decoder_output_names = ["logits", "output_past_key_values"]

        dyn_axis_general = {0: "batch", 1: "sequence"}
        dyn_axis_pkv = {0: "batch", 2: "seq_length"}

        dyn_axis = {
            "input_ids": dyn_axis_general,
            "encoder_attention_mask": dyn_axis_general,
            "encoder_hidden_states": dyn_axis_general,
            "logits": dyn_axis_general,
            "output_past_key_values": dyn_axis_general,
        }

        dyn_pkv = {
            "pkv_{}".format(i): dyn_axis_pkv
            for i in range(len(flat_past_key_values))
        }

        dyn_axis_params = {**dyn_axis, **dyn_pkv}

        # decoder to utilize past key values:
        torch.onnx.export(
            decoder_with_lm_head,
            decoder_all_inputs,
            decoder_path.as_posix(),
            export_params=True,
            do_constant_folding=True,
            opset_version=onnx_opset_version,
            input_names=decoder_input_names,
            output_names=decoder_output_names,
            dynamic_axes=dyn_axis_params,
        )
        bar.next()

        torch.onnx.export(
            simplified_encoder,
            args=(encoder_input_ids, encoder_attention_mask),
            f=encoder_path.as_posix(),
            export_params=True,
            opset_version=onnx_opset_version,
            do_constant_folding=True,
            input_names=["input_ids", "attention_mask"],
            output_names=["hidden_states"],
            dynamic_axes={
                "input_ids": dyn_axis_general,
                "attention_mask": dyn_axis_general,
                "hidden_states": dyn_axis_general,
            },
        )
        bar.next()

        # initial decoder to produce past key values
        torch.onnx.export(
            decoder_with_lm_head_init,
            (decoder_input_ids, decoder_attention_mask, encoder_output),
            init_decoder_path.as_posix(),
            export_params=True,
            opset_version=onnx_opset_version,
            input_names=[
                "input_ids",
                "encoder_attention_mask",
                "encoder_hidden_states",
            ],
            output_names=["logits", "past_key_values"],
            dynamic_axes={
                # batch_size, seq_length = input_shape
                "input_ids": dyn_axis_general,
                "encoder_attention_mask": dyn_axis_general,
                "encoder_hidden_states": dyn_axis_general,
                "logits": dyn_axis_general,
                "past_key_values": dyn_axis_general,
            },
        )
        bar.next()
        bar.finish()

    return encoder_path, decoder_path, init_decoder_path

def get_model_paths(model_checkpoint: str = None, 
                    model_path: Path = None, 
                    quantized=True):

    model_path.mkdir(parents=True, exist_ok=True)

    # gets only the filename
    model_checkpoint_name = Path(model_checkpoint).stem

    if not quantized:
        encoder_path = model_path.joinpath(f"{model_checkpoint_name}-encoder.onnx")
        decoder_path = model_path.joinpath(f"{model_checkpoint_name}-decoder.onnx")
        init_decoder_path = model_path.joinpath(
            f"{model_checkpoint_name}-init-decoder.onnx"
        )
    else:
        encoder_path = model_path.joinpath(
            f"{model_checkpoint_name}-encoder-quantized.onnx"
        )
        decoder_path = model_path.joinpath(
            f"{model_checkpoint_name}-decoder-quantized.onnx"
        )
        init_decoder_path = model_path.joinpath(
            f"{model_checkpoint_name}-init-decoder-quantized.onnx"
        )

    return encoder_path, decoder_path, init_decoder_path

def quantize(onnx_model_paths: Iterable[Path] = None):
    """
    https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/quantize.py
    """

    bar = Bar("Quantizing...", max=3)

    quant_model_paths = []
    for model in onnx_model_paths:
        model_name = model.as_posix()
        output_model_name = f"{model_name[:-5]}-quantized.onnx"
        quantize_dynamic(
            model_input=model_name,
            model_output=output_model_name,
            per_channel=True,
            reduce_range=True, # should be the same as per_channel
            weight_type=QuantType.QInt8,  # per docs, signed is faster on most CPUs
            optimize_model=False,
        )  # op_types_to_quantize=['MatMul', 'Relu', 'Add', 'Mul' ],
        quant_model_paths.append(output_model_name)
        bar.next()
    
    bar.finish()

    return tuple(quant_model_paths)

if __name__ == '__main__':

    pegasus_checkpoint = 'Yale-LILY/brio-xsum-cased' 
    bart_checkpoint = 'Yale-LILY/brio-cnndm-uncased'

    pegasus_output = '../models/pegasus'
    bart_output = '../models/bart'

    pegasus_onnx_paths = export_to_onnx(pegasus_checkpoint, output_path=pegasus_output)
    bart_onnx_paths = export_to_onnx(bart_checkpoint, output_path=bart_output)
    quantized_pegasus_paths = quantize(pegasus_onnx_paths)
    quantized_bart_paths = quantize(bart_onnx_paths)

    print(f"Quantized BRIO-Pegasus ONNX models at : {quantized_pegasus_paths}")
    print(f"Quantized BRIO-Bart models at : {quantized_bart_paths}")



