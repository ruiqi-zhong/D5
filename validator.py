import pickle as pkl
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer
from typing import List, Dict
from tqdm import trange

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 32
sm = torch.nn.Softmax(dim=-1)
MAX_TARGET_LENGTH = 2
YES_NO_TOK_IDX = [150, 4273]
MAX_SOURCE_LENGTH = 1024
TEMPERATURE = 0.001

# if you have multiple GPUs, you can parallelize the T5 model across them
# future packages version will probably have more efficient ways to do this
def parallelize_across_device(model):
    num_heads = len(model.encoder.block)
    num_device = torch.cuda.device_count()
    other_device_alloc = num_heads // num_device + 1
    first_device = num_heads - (num_device - 1) * other_device_alloc
    device_map = {}
    cur = 0
    end = max(cur + first_device, 1)
    device_map[0] = list(range(cur, end))
    cur = end
    for i in range(1, num_device):
        end = min(cur + other_device_alloc, num_heads)
        device_map[i] = list(range(cur, end))
        cur += other_device_alloc
    print('device_map', device_map)
    model.parallelize(device_map)


DEFAULT_VALIDATOR_TEMPLATE = open('templates/t5_validator.txt', 'r').read()
class Validator:

    # model_path is the path to the T5 model weights used for validation
    # can also any other model name
    # the default is the best model we have trained
    def __init__(self, model_path: str ='ruiqi-zhong/d5_t5_validator', batch_size: int = BATCH_SIZE, verbose: bool = False, template: str = DEFAULT_VALIDATOR_TEMPLATE):
        self.tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
        print('loading model weights')
        self.model = T5ForConditionalGeneration.from_pretrained(model_path)
        print('done')
        parallelize_across_device(self.model)
        self.validator_template = template
        self.batch_size = batch_size
        self.verbose = verbose
    
    # input_dicts is a list of dictionaries, each dictionary has two keys: "hypothesis" (h) and "text" (x), mapping to the hypothesis and text to be validated
    # returns a list of scores, each score is a float between 0 and 1, corresponding to the probability that the hypothesis is true given the text for each input dictionary
    # note that it is an iterator, so you can use it in a for loop and save the results whenever some input dictionaries are processed
    def validate_w_scores(self, input_dicts: List[Dict[str, str]]) -> List[float]:
        prompts = []
        for i, input_dict in enumerate(input_dicts):
            hypothesis, text = input_dict['hypothesis'], input_dict['text']
            prompts.append(self.validator_template.format(hypothesis=hypothesis, text=text))
        
        with torch.no_grad():
            self.model.eval()
            num_batches = (len(prompts) - 1) // self.batch_size + 1
            if self.verbose:
                pbar = trange(num_batches)
                pbar.set_description('inference')
            else:
                pbar = range(num_batches)

            for batch_idx in pbar:
                input_prompts = prompts[batch_idx * self.batch_size : (batch_idx + 1) * self.batch_size ]
                inputs = self.tokenizer(input_prompts, 
                                    return_tensors="pt",
                                    padding="longest",
                                    max_length=MAX_SOURCE_LENGTH,
                                    truncation=True,
                                    ).to(device)
                generation_result = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    do_sample=True,
                    temperature=TEMPERATURE,
                    max_new_tokens=MAX_TARGET_LENGTH,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                scores = sm(generation_result.scores[0][:,YES_NO_TOK_IDX])[:,1].detach().cpu().numpy().tolist()
                for s in scores:
                    yield s

class DummyValidator:

    def __init__(self):
        print('!!!!! WARNING: You are using a dummy verifier that returns random results!!!!!!!')
        pass
    
    def validate_w_scores(self, ind_dicts):
        for _ in range(len(ind_dicts)):
            yield 0.01


if __name__ == '__main__':
    input_dicts = [ 
        {'hypothesis': 'is a positive review', 'text': 'I like this movie.'}, 
        {'hypothesis': 'is a positive review', 'text': 'I hate this movie.'}
    ]
    input_dicts = input_dicts * 100
    validator = Validator('../workflow/mount/models/0221distill_verifier_google-flan-t5-xl/checkpoint-10000/')
    all_results = []
    for s in validator.validate_w_scores(input_dicts):
        all_results.append(s)
    import numpy as np
    all_results = np.array(all_results)
    pred = (all_results > 0.5).astype(int)
    gold = np.array([1, 0] * 100)
    print('acc', (pred == gold).mean())
