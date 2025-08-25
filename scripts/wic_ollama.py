from ollama import chat
from wic_utils import get_prompt

def llm_for_wic(lemma: str,
                sentence1: str,
                sentence2: str,
                model: str = 'qwen3:4b',
                seed: int = 9999,
                no_think: bool = False) -> bool:
    try:
        prompt = get_prompt(lemma, sentence1, sentence2, no_think)
        response = chat(model=model,
                        options={
                            'num_predict':10,
                            'temperature':0.0,
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
    
