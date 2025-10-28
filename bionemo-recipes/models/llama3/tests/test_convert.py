from transformers import AutoModelForCausalLM

from convert import convert_llama_hf_to_te


def test_convert_llama_hf_to_te():
    model_hf = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    model_te = convert_llama_hf_to_te(model_hf)

    assert model_te is not None
    breakpoint()
