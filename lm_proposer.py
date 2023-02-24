
from transformers import AutoTokenizer
import nltk
import openai
import pickle as pkl
import random
import numpy as np
from copy import deepcopy
import os
from typing import Dict, List
import time

openai.api_key = os.environ['openai_key']


GPT3_TOK = AutoTokenizer.from_pretrained('gpt2-medium')
SINGLE_SAMPLE_MAX_LENGTH = 256
MAX_PROMPT_LENGTH = 3200
SLEEP_TIME_AFTER_ERROR = 20


# the default hypotheses that we use for the prompt
# if we do not use problem-specific hypotheses
DEFAULT_HYPOTHESES = [
    "talks about politics, such as presidential election.",
    "contains insulting language for immigrants.",
    "uses double negation, i.e., using two negations in a sentence."
]


# classify whether a text sample x is a comparison
def classify_cmp(x: str) -> bool:
    tokenized_x = nltk.word_tokenize(x)
    pos_tags = nltk.pos_tag(tokenized_x)
    all_tags = {t[1] for t in pos_tags}
    return any(tag in ('JJR', 'RBR') for tag in all_tags)


# construct components that will be used for the prompt
def construct_blocks(A_samples: List[str], B_samples: List[str], num_incontext_samples: int = 25):
    A_subsampled_samples = np.random.choice(A_samples, min(num_incontext_samples, len(A_samples)), replace=False)
    A_block = ''.join(['Group A: ' + s + '\n' for s in A_subsampled_samples])

    B_subsampled_samples = np.random.choice(B_samples, min(num_incontext_samples, len(B_samples)), replace=False)
    B_block = ''.join(['Group B: ' + s + '\n' for s in B_subsampled_samples])

    return {
        'A_block': A_block,
        'B_block': B_block,
        'A_subsampled_samples': A_subsampled_samples,
        'B_subsampled_samples': B_subsampled_samples
    }


# truncate a text sample x to a maximum length
def prefix_subspan(x: str, prefix_token_max_len: int = SINGLE_SAMPLE_MAX_LENGTH, tok: AutoTokenizer = GPT3_TOK) -> str:
    tokens = tok.tokenize(x)
    total_length = len(tokens)
    if total_length <= prefix_token_max_len:
        return x
    subspan_toks = tokens[:prefix_token_max_len]
    return tok.convert_tokens_to_string(subspan_toks) + '...'


rm_cmp_prompt = open('templates/rm_cmp_prompt.txt').read()
# remove comparison from a text sample x
def convert_cmp_to_ind(s: str) -> str:
    for _ in range(3):
        if not classify_cmp(s):
            break
        prompt = rm_cmp_prompt.format(input=s)
        response = gpt3wrapper(prompt=prompt, max_tokens=2048, temperature=0.0, top_p=1, frequency_penalty=0.0, presence_penalty=0.0, stop=['\n\n'], engine='text-davinci-002')
        if response is None:
            return s
        s = response['choices'][0]['text'].strip()
    if classify_cmp(s) or 'group a' in s.lower() or 'group b' in s.lower():
        return None
    return s

# a wrapper for openai.Completion.create
# if the API call fails, it will retry for max_repeat times
def gpt3wrapper(max_repeat=20, **arguments):
    i = 0
    while i < max_repeat:
        try:
            response = openai.Completion.create(**arguments)
            return response
        except KeyboardInterrupt:
            raise KeyboardInterrupt
        except Exception as e:
            print(e)
            print(arguments['prompt'])
            print('now sleeping for %d seconds...' % SLEEP_TIME_AFTER_ERROR)
            time.sleep(SLEEP_TIME_AFTER_ERROR)
            i += 1
    return None


class GPT3_Proposer:

    def __init__(self, problem, use_default_hypotheses=False, single_max_length=SINGLE_SAMPLE_MAX_LENGTH, engine_name='text-davinci-003', temperature=0.7):
        if use_default_hypotheses:
            self.example_hypotheses = DEFAULT_HYPOTHESES
        else:
            self.example_hypotheses = (problem['example_hypotheses'] + DEFAULT_HYPOTHESES)[:3]
        
        self.problem = problem
        self.prompt_template = open('templates/gpt3_proposer.txt', 'r').read()
        self.single_max_length = single_max_length
        self.engine_name = engine_name
        self.temperature = temperature
    
    def propose_hypotheses(self, X_A: List[str], X_B: List[str]):
        X_A = [prefix_subspan(x) for x in X_A]
        X_B = [prefix_subspan(x) for x in X_B]

        num_incontext_samples = 25
        prompt = None
        arg_dict = {
            k: self.problem[k] for k in ['dataset_description', 'generation', 'A_desc', 'B_desc', 'user', 'target']
        }
        random.shuffle(self.example_hypotheses)
        for i, hypothesis in enumerate(self.example_hypotheses):
            arg_dict[f'example_hypothesis_{i+1}'] = hypothesis

        while num_incontext_samples > 1:

            sent_subset = construct_blocks(X_A, X_B, num_incontext_samples=num_incontext_samples)
            
            A_block, B_block = sent_subset['A_block'], sent_subset['B_block']
            tmp_arg_dict = deepcopy(arg_dict)
            tmp_arg_dict['A_block'] = A_block
            tmp_arg_dict['B_block'] = B_block
            prompt = self.prompt_template.format(**tmp_arg_dict)
            prompt_length = len(GPT3_TOK.encode(prompt))
            if prompt_length < MAX_PROMPT_LENGTH:
                break
            else:
                # prompt too long, reducing num_incontext_samples
                print(num_incontext_samples)
                num_incontext_samples -= 1

        arg_dict['A_block'] = sent_subset['A_block']
        arg_dict['B_block'] = sent_subset['B_block']
        prompt = self.prompt_template.format(**arg_dict)

        query_args = {
            'engine': self.engine_name,
            'prompt': prompt,
            'temperature': self.temperature,
            'max_tokens': 512,
            'top_p': 1,
            'n': 1
        }

        result = gpt3wrapper(**query_args)

        returned_text = result['choices'][0]['text']

        hs = []
        for h in returned_text.split('\n\n')[0].split('\n-'):
            h = convert_cmp_to_ind(h.replace('"', '').strip())
            if h is not None and len(h) > 0:
                if h[-1] == '.':
                    h = h[:-1]
                hs.append(h)

        return {
            'hypotheses': hs,
            'query_args': query_args
        }


def flip_problem(problem):
    problem = deepcopy(problem)
    problem['A_desc'], problem['B_desc'] = problem['B_desc'], problem['A_desc']
    problem['split'] = {
        k: {
            'A_samples': v['B_samples'],
            'B_samples': v['A_samples']
        } for k, v in problem['split'].items()
    }
    return problem


if __name__ == '__main__':
    test_function = True
    if test_function:
        assert not classify_cmp('He is a good person.')
        assert classify_cmp('I can do better than that')
        print('====================')
        s = 'contains more refereces to the "good" side'
        print('original hypotheses: ', s)
        print('removed comparatives: ', convert_cmp_to_ind(s))
        print('====================')
        s = ';'.join('a long string %d' % i for i in range(100))
        shorter_str = prefix_subspan(s)
        print('this string should be truncated: ', shorter_str)
        print('====================')
        A_samples = ['A sample %d' % i for i in range(100)]
        B_samples = ['B sample %d' % i for i in range(100)]
        sent_subset = construct_blocks(A_samples, B_samples, num_incontext_samples=25)
        print('A block: ', sent_subset['A_block'])

    example_problem = pkl.load(open('example_problem.pkl', 'rb'))

    gpt3_proposer = GPT3_Proposer(example_problem)
    A_samples = example_problem['split']['research']['A_samples']
    B_samples = example_problem['split']['research']['B_samples']
    result = gpt3_proposer.propose_hypotheses(A_samples, B_samples)

    print("Here's the prompt used to propose hypotheses:")
    prompt_used = result['query_args']['prompt']
    print(prompt_used)

    print("Here's some example hypotheses:")
    for i, h in enumerate(result['hypotheses'][:5]):
        print('hypothesis %d:' % i, h)