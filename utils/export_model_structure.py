import torch

class BrioEncoder(torch.nn.Module):
    """Encoder to output only hidden states
    """

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, *input, **kwargs):
        return self.encoder(*input, **kwargs)[0]

class BrioDecoderLMHead(torch.nn.Module):
    """Decoder with language modeling head to utilize past key values
    """

    def __init__(self, decoder, lm_head, final_logits_bias, config):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.final_logits_bias = final_logits_bias
        self.config = config
    
    def forward(self, input_ids, attention_mask, encoder_hidden_states, *pkv):

        # convert flattened past_key_values inputs to tuples that the transformer decoder expects
        past_key_values = tuple(pkv[i : i + 4] for i in range(0, len(pkv), 4))

        decoder_output = self.decoder(
            input_ids=input_ids,  
            encoder_attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            past_key_values=past_key_values,
        )

        lm_head_out = self.lm_head(decoder_output[0]) + self.final_logits_bias

        return lm_head_out, decoder_output[1]

class BrioDecoderLMHeadInitial(torch.nn.Module):
    """Initial decoder with language modeling head to produce past key values
    """

    def __init__(self, decoder, lm_head, final_logits_bias, config):
        super().__init__()
        self.decoder = decoder
        self.lm_head = lm_head
        self.final_logits_bias = final_logits_bias
        self.config = config

    def forward(self, input_ids, attention_mask, encoder_hidden_states):
        decoder_output = self.decoder(
            input_ids=input_ids,
            encoder_attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
        )

        return self.lm_head(decoder_output[0]) + self.final_logits_bias, decoder_output[1]

