import numpy as np
import random
import tqdm
from typing import List
import scipy.stats as stats
from validator import Validator
from lm_proposer import GPT3_Proposer
from sample_lexical_diversity import lexical_diversity
from scipy.stats import norm


def calculate_diff_w_significance(A_scores, B_scores, alpha=1e-5):
    A_scores = np.array(A_scores)
    B_scores = np.array(B_scores)
    mu = np.mean(A_scores) - np.mean(B_scores)
    p_value = stats.ttest_ind(A_scores, B_scores, alternative='greater')[1]
    mu_variance = np.var(A_scores) / len(A_scores) + np.var(B_scores) / len(B_scores)
    mu_std = np.sqrt(mu_variance)
    target_z = norm.ppf(1 - alpha / 2)
    lo, hi = mu - target_z * mu_std, mu + target_z * mu_std
    return {
        'mu': mu,
        'p_value': p_value,
        'mu_std': mu_std,
        'lo': lo,
        'hi': hi
    }


DEBUG = False
eps = 1e-5
VALIDATE_HYP_BLOCK_SIZE = 32

class D5:

    def __init__(
        self,
        A_samples: List[str], # A_samples are the research split of Corpus A, (optionally) sorted by its representativeness. Corresponds to D^{res}_{A} in the paper
        B_samples: List[str], # B_samples are the research split of Corpus B, (optionally) sorted by its representativeness. Corresponds to D^{res}_{B} in the paper
        validator: Validator, # validator can map (hypothesis, sample) to a score (corresponding to T'(h, x) in the paper)
        proposer, # proposer can propose a list of hypotheses given a list of samples from Corpus A and Corpus B
        top_fraction: List[float] = None, # by default we use 0.05, 0.2, 1.0
        total_hypotheses_count: int = 60, # the size of H_init
        early_stop: bool = True, # whether not to validate the unpromising hypotheses on more samples
        top_K_hypotheses: int = 5 # the size of H_final
    ):
        self.A_samples, self.B_samples = A_samples, B_samples
        self.proposer, self.validator = proposer, validator

        # whether a sample belongs to Corpus A (1) or Corpus B (0)
        self.sample2membership = {}
        for sample in A_samples:
            self.sample2membership[sample] = 1.
        for sample in B_samples:
            self.sample2membership[sample] = 0.

        # maintain a mapping from a hypothesis (string) to a dictionary of information
        self.h2h_dicts = {}

        # other hyperparameters
        self.top_fraction = top_fraction
        if top_fraction is None:
            self.top_fraction = [0.05, 0.2, 1.0]
        self.total_hypotheses_count = total_hypotheses_count
        self.early_stop = early_stop
        self.top_K_hypotheses = top_K_hypotheses


    def get_hypotheses(self):

        for idx in range(3):
            for p in self.top_fraction:
                if len(self.h2h_dicts) >= self.total_hypotheses_count:
                    break
                X_A, X_B = lexical_diversity(self.A_samples, self.B_samples, top_p=p, num_samples=25)
                r = self.proposer.propose_hypotheses(X_A, X_B)
                hyps, provenance = r['hypotheses'], r['query_args']
                provenance['top_p'] = p
                provenance['idx'] = idx
                for hyp in hyps:
                    if hyp not in self.h2h_dicts and len(self.h2h_dicts) < self.total_hypotheses_count:
                        h_dict = {
                            'hypothesis': hyp,
                            'sample2score': {}, 
                            'provenance': provenance,
                            'diff_w_significance': None,
                            'active': True
                        }
                        self.h2h_dicts[hyp] = h_dict

    def get_V_info(self):
        for h in self.h2h_dicts:
            hyp_dict = self.h2h_dicts[h]
            ordered_text = sorted(hyp_dict['sample2score'], key=hyp_dict['sample2score'].get)

            A_scores = [hyp_dict['sample2score'][sample] for sample in ordered_text if self.sample2membership[sample] == 1.]
            B_scores = [hyp_dict['sample2score'][sample] for sample in ordered_text if self.sample2membership[sample] == 0.]
            self.h2h_dicts[h]['diff_w_significance'] = calculate_diff_w_significance(A_scores, B_scores)
    

    def filter_weak_hypotheses(self):
        # obtain the lower bounds of the confidence intervals
        lower_bounds = [hyp_dict['diff_w_significance']['lo'] for hyp_dict in self.h2h_dicts.values()]
        # if early stopping, we only consider the top K hypotheses
        threshold = sorted(lower_bounds, reverse=True)[:self.top_K_hypotheses][-1] if self.early_stop else 0

        for h, hyp_dict in self.h2h_dicts.items():
            if hyp_dict['active'] and hyp_dict['diff_w_significance']['hi'] < threshold:
                hyp_dict['active'] = False


    def validate(self):
        random_sample_order = list(self.sample2membership.keys())
        random.shuffle(random_sample_order)

        cur_pointer = 0

        print('Filtering out weak hypotheses')
        # enumerate the samples in random order
        with tqdm.tqdm(total=len(random_sample_order)) as pbar:
            while cur_pointer < len(random_sample_order):

                # take a batch of samples, and compute a score for every competitive hypotheses
                samples = random_sample_order[cur_pointer:cur_pointer+VALIDATE_HYP_BLOCK_SIZE]
                cur_pointer += VALIDATE_HYP_BLOCK_SIZE

                # construct the validator dicts
                validator_dicts = []
                for sample in samples:
                    for h, hyp_dict in self.h2h_dicts.items():
                        if not hyp_dict['active']:
                            continue
                        validator_dict = {'hypothesis': h, 'text': sample, 'pointer': hyp_dict}
                        validator_dicts.append(validator_dict)
                
                # run the validator 
                all_scores = list(self.validator.validate_w_scores(validator_dicts))
                assert len(all_scores) == len(validator_dicts)
                for d, s in zip(validator_dicts, all_scores):
                    # add small perturbation so that the spearmanr correlation is still well-defined
                    d['pointer']['sample2score'][d['text']] = s + eps * random.random()
                
                # filter out weaker hypotheses based on UCB
                pbar.update(len(samples))
                self.get_V_info()
                self.filter_weak_hypotheses()

                pbar.set_description('Num hypotheses: %d' % len([h for h in self.h2h_dicts if self.h2h_dicts[h]['active']]))

    def run(self):

        # obtain H_init, the hypotheses are stored in self.h2h_dicts
        self.get_hypotheses()

        # validate the hypotheses
        # the validation results are in self.h2h_dicts[h]['sample2score']
        self.validate()

        # compute how valid each hypothesis is
        self.get_V_info()

        return self.h2h_dicts


if __name__ == '__main__':
    from validator import DummyValidator
    import pickle as pkl

    problem = pkl.load(open('example_problem.pkl', 'rb'))

    v = DummyValidator()
    proposer = GPT3_Proposer(problem)
    A_samples = problem['split']['research']['A_samples']
    B_samples = problem['split']['research']['B_samples']

    d5 = D5(A_samples, B_samples, v, proposer, total_hypotheses_count=100, top_fraction=[0.05, 0.2, 1.0], early_stop=True)
    h2h_dicts = d5.run()