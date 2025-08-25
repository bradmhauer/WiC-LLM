from transformers import pipeline, set_seed
from wic_utils import get_prompt

# Suppress all transformers warnings and logging
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Optional: avoid tokenizers warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="transformers")
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
import logging
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.tokenization_utils_base").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)

pipe = None

def llm_for_wic(lemma: str,
                sentence1: str,
                sentence2: str,
                model: str = 'Qwen/Qwen3-1.7B',
                seed: int = 9999,
                no_think: bool = False,) -> bool:

    global pipe
    if pipe is None:
        pipe = pipeline('text-generation', model=model)

    set_seed(seed)
        
    # Define generation parameters for greedy sampling.
    generation_kwargs = {
        "max_new_tokens": 1024,
        "temperature": 0.01,
        "top_k": 1,
        "top_p": 1.0,
        "do_sample": False,
        "repetition_penalty": 1.0,
    }
        
    try:
        prompt = get_prompt(lemma, sentence1, sentence2, no_think)
        messages = [{'role':'user', 'content':prompt}]
        pipe_out = pipe(messages, **generation_kwargs)
        content  = pipe_out[0]['generated_text'][-1]['content']

        parts = content.split('</think>')
        if len(parts) >= 2:
            answer = parts[1].strip()
        else:
            answer = content.strip()

        return(answer.lower().startswith('same'))

    except Exception as e:
        print(f"Error in LLM processing: {e}")
        return False


if __name__ == '__main__':
    lem = 'bank'
    s1 = 'Under the bridge on the bank of the river.'
    s2 = 'I have to go to the bank to deposit some cash.'
    s3 = 'The river bank was slippery after the rain.'
    s4 = 'Is the bank open this late?'
    sentences = [s1,s2,s3,s4]

    for si in sentences:
        for sj in sentences:
            print(si)
            print(sj)
            print(llm_for_wic(lem, si, sj, no_think=True))
            print()
    
