from ollama import chat
from wic_utils import get_prompt

def llm_for_wic(lemma: str,
                sentence1: str,
                sentence2: str,
                model: str = 'qwen3:1.7b-fp16',
                seed: int = 9999,
                no_think: bool = True) -> bool:
    try:
        prompt = get_prompt(lemma, sentence1, sentence2, no_think)
        response = chat(model=model,
                        options={
                            'num_predict':10,
                            'temperature':0.00000001,
                            'top_k':1,
                            'top_p':1.0,
                            'seed':seed,
                        },
                        messages=[{
                            'role': 'user',
                            'content': prompt,
                        }],
                        )

        # Split on '</think>' and handle potential IndexError
        content = response.message.content
        parts = content.split('</think>')
        if len(parts) >= 2:
            answer = parts[1].strip()
        else:
            # Fallback if splitting fails
            answer = content.strip()

        return(answer.lower().startswith('same'))

    except Exception as e:
        print(f"Error in LLM processing: {e}")
        return(False)


if __name__ == '__main__':
    lemma = "bank"
    s1 = "Under the bridge on the bank of the river."
    s2 = "I have to go to the bank to deposit some cash."
    s3 = "Is the bank still open this late?."

    print(s1)
    print(s2)
    print(llm_for_wic(lemma, s1, s2, model="qwen3:1.7b-fp16", seed=9999)) # False
    print()
    print(s2)
    print(s3)
    print(llm_for_wic(lemma, s2, s3, model="qwen3:1.7b-fp16", seed=9999)) # True
