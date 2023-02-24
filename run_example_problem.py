import pickle as pkl
import random
from copy import deepcopy
from tqdm import tqdm
import os
import time
import sys

from D5 import D5
from validator import DummyValidator, Validator
from lm_proposer import GPT3_Proposer
from get_representative import return_extreme_values
import argparse


def subsample(samples, n=1000):
    selected_idxes = list(range(len(samples)))
    random.shuffle(selected_idxes)
    selected_idxes = selected_idxes[:n]
    return [samples[i] for i in sorted(selected_idxes)]

# whether to run a proof-of-concept demo
# if not we will run the entire pipeline that takes a longer time to run but produces better results
demo = True if len(sys.argv) > 1 and sys.argv[1] == 'demo' else False
use_subsample = True
find_representative = False if demo else True
use_dummy_verifier = True if demo else False



if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--find_representative', default=False, action='store_true',
        help='whether to find the representative samples from each corpus to prompt the proposer. If False, we will randomly select samples to prompt the proposer.'
    )
    parser.add_argument(
        '--subsample', type=int, default=None,
        help='only consider <subsample> samples from each corpus to run faster.'
    )
    parser.add_argument(
        '--verifier_name', type=str, default='ruiqi-zhong/d5_t5_validator', 
        choices=['dummy', 'ruiqi-zhong/d5_t5_validator', 'ruiqi-zhong/d5_t5_validator_700M', 'ruiqi-zhong/d5_t5_validator_3B'],
        help='The name of the verifier to use. If dummy, use a dummy verifier that returns random results. ruiqi-zhong/d5_t5_validator is the best model we have trained, but it is large. ruiqi-zhong/d5_t5_validator_700M and ruiqi-zhong/d5_t5_validator_3B are smaller distilled models that are faster to run but produce slightly worse results; however, they should still be able to perform easier tasks like classifying topics.'
    )
    parser.add_argument(
        '--verifier_batch_size', type=int, default=32, 
        help='The batch size to use for the verifier. Decrease it if you are running out of memory.'
    )
    parser.add_argument(
        '--problem_path', type=str, default='example_problem.pkl', 
        help='The path to the problem pickle file. You can also use your own problem pickle file.'
    )
    parser.add_argument(
        '--output_path', type=str, default='output.pkl', 
        help='The path to save the output pickle file. You can also use your own output pickle file.'
    )

    args = parser.parse_args()

    # loading the problem from the pickle file
    problem = pkl.load(open(args.problem_path, 'rb'))

    # finding the representative samples from each corpus in the problem
    # you can comment it out if you want to save time
    if args.find_representative:
        extreme_vals = return_extreme_values(problem['split']['research']['A_samples'], problem['split']['research']['B_samples'])
        problem['split']['research']['A_samples'], problem['split']['research']['B_samples'] = extreme_vals['sorted_A_samples'], extreme_vals['sorted_B_samples']

    # subsampling the representative samples
    # since verifying the hypotheses is expensive, we only verify a smaller subset of the samples
    # you can comment it out if you want to save time
    if args.subsample:
        problem['split']['research']['A_samples'], problem['split']['research']['B_samples'] = subsample(problem['split']['research']['A_samples'], args.subsample), subsample(problem['split']['research']['B_samples'], args.subsample)

    # creating the proposer and verifier
    proposer = GPT3_Proposer(problem)

    # for actual use, the verifier is a validator with 11B parameters
    # for debugging, the verifier is a dummy validator returns a random value
    if args.verifier_name == 'dummy':
        verifier = DummyValidator()
    else:
        verifier = Validator(args.verifier_name, batch_size=args.verifier_batch_size)

    # goal-driven discovery and description of corpus-level differences
    d5 = D5(
        problem['split']['research']['A_samples'], 
        problem['split']['research']['B_samples'], 
        verifier,
        proposer,
        total_hypotheses_count=60,
        early_stop=True
    )
    h2h_dicts = d5.run()

    h_sorted = sorted(h2h_dicts, key=lambda h: h2h_dicts[h]['diff_w_significance']['mu'], reverse=True)
    for h in h_sorted:
        h_dict = h2h_dicts[h]
        # print out the example hypothesis along with their V' score
        print(h_dict['hypothesis'], 'V\'', h_dict['diff_w_significance']['mu'])
    pkl.dump(h2h_dicts, open(args.output_path, 'wb'))