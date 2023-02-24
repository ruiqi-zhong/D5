import random
from copy import deepcopy
import pickle as pkl
from tqdm import tqdm
import os
import time


def subsample(samples, n=1000):
    selected_idxes = list(range(len(samples)))
    random.shuffle(selected_idxes)
    selected_idxes = selected_idxes[:n]
    return [samples[i] for i in sorted(selected_idxes)]

# flip the problem of describing corpus A to describing corpus B
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
    from D5 import D5
    from validator import DummyValidator, Validator
    from lm_proposer import GPT3_Proposer
    from get_representative import return_extreme_values

    problems = pkl.load(open('OpenD5.pkl', 'rb'))

    # the default validator has 11B parameters
    validator = Validator()
    if not os.path.exists('discoveries'):
        os.mkdir('discoveries')

    # randomly shuffle the problems to run our system on
    problem_idxes = list(range(len(problems)))
    random.shuffle(problem_idxes)
    pbar = tqdm(problem_idxes)


    for problem_id in pbar:
        pbar.set_description(f'problem {problem_id}')
        problem_orig = problems[problem_id]

        def get_h2h_dicts(problem, save_path):
            current_time = time.time()
            if os.path.exists(save_path):
                loaded_object = pkl.load(open(save_path, 'rb'))
                if type(loaded_object) == dict:
                    return loaded_object
                elif current_time - loaded_object < 60 * 60 * 8:
                    return None
            pkl.dump(current_time, open(save_path, 'wb'))

            extreme_vals = return_extreme_values(problem['split']['research']['A_samples'], problem['split']['research']['B_samples'])
            problem['split']['research']['A_samples'] = subsample(extreme_vals['sorted_A_samples'])
            problem['split']['research']['B_samples'] = subsample(extreme_vals['sorted_B_samples'])

            proposer = GPT3_Proposer(problem)
            d5 = D5(
                problem['split']['research']['A_samples'], 
                problem['split']['research']['B_samples'], 
                validator,
                proposer,
                total_hypotheses_count=60,
                early_stop=True
            )
            h2h_dicts = d5.run()
            pkl.dump(h2h_dicts, open(save_path, 'wb'))
            return h2h_dicts
        
        get_h2h_dicts(deepcopy(problem_orig), f'discoveries/additional_describe_A_{problem_id}.pkl')
        if problem_orig['flip']:
            get_h2h_dicts(flip_problem(deepcopy(problem_orig)), f'discoveries/additional_describe_B_{problem_id}.pkl')


