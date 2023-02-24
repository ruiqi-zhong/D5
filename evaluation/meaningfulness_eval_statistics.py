import pandas as pd
from scipy.stats import pearsonr, spearmanr, ttest_rel
from collections import defaultdict
import numpy as np
from sklearn.metrics import cohen_kappa_score

names = ['author%d' % i for i in range(3)]
evaluation_df = pd.read_csv('evaluation/meaningfulness_eval.csv')
print('Here are the columns of the dataframe')
print(evaluation_df.columns)
print('In total we evaluated %d hypotheses' % len(evaluation_df))
print('Author_0\'s rating of relevance has the column name of author0_rel')

names = ['author0', 'author1', 'author2']
metrics = ['rel', 'nov', 'sig']

# for each metric, compute the average rating across all evaluators
for metric in metrics:
    evaluation_df['avg_' + metric] = np.mean([evaluation_df[name + '_' + metric].values for name in names], axis=0)


ds = {}
##################### calculate the averaged rating for each metric ##################### 
w_context_idxes = evaluation_df['w_context'] == True
for metric in metrics:
    d = {
        'with_context': np.mean([evaluation_df[w_context_idxes][name + '_' + metric] for name in names]),
        'without_context': np.mean([evaluation_df[~w_context_idxes][name + '_' + metric] for name in names])
    }
    ds[metric] = d

##################### average all pairwise evaluator agreement rate for each metric ##################### 
metric2agreement = {}
for metric in metrics:
    all_kappa = []
    all_spearman = []
    for n1 in names:
        for n2 in names:
            if n1 < n2:
                all_kappa.append(cohen_kappa_score(evaluation_df[n1 + '_' + metric], evaluation_df[n2 + '_' + metric]))
                all_spearman.append(spearmanr(evaluation_df[n1 + '_' + metric], evaluation_df[n2 + '_' + metric])[0])
    ds[metric].update({
        'kappa': np.mean(all_kappa),
        'spearmanr': np.mean(all_spearman)
    })
    
displayed_df = pd.DataFrame(ds)
print('average rating of using and not using context; and interannotator agreement rate')
print(displayed_df.round(2).T.to_latex())


##################### for each metric, use paired t-test to compute the p-value for the null hypothesis that the averaged rating is the same ##################### 
##################### each pair is (the averaged rating of the hypotheses on a problem before applying context) vs. (... after applying context) ##################### 
print('=======================================================')
print('p-value that using context and not using context leads to the same averaged rating')
idxes = list(set(evaluation_df['problem_id']))
for metric in metrics:

    df = evaluation_df[evaluation_df['w_context'] == True]
    application_ids = df['problem_id']
    id2scores = {i: [] for i in idxes}
    for name in names:
        scores = df[name + '_' + metric]
        for i, s in zip(application_ids, scores):
            id2scores[i].append(s)
    s_w_context = [np.mean(id2scores[i]) for i in idxes]

    df = evaluation_df[evaluation_df['w_context'] != True]
    application_ids = df['problem_id']
    id2scores = {i: [] for i in idxes}
    for name in names:
        scores = df[name + '_' + metric]
        for i, s in zip(application_ids, scores):
            id2scores[i].append(s)
    s_wout_context = [np.mean(id2scores[i]) for i in idxes]
    
    p = ttest_rel(s_w_context, s_wout_context)[1]
    print(metric, p)

##################### for each metric, test the null that using context and not using context lead to the same rating on individual evaluators ##################### 
##################### we report the largest p-value here ##################### 
print('=======================================================')
print('Largest p-value across evaluators')
idxes = list(set(evaluation_df['problem_id']))
for metric in metrics:
    least_sig = 0
    for name in names:
        df = evaluation_df[evaluation_df['w_context'] == True]
        application_ids = df['problem_id']
        scores = df[name + '_' + metric]
        id2scores = {i: [] for i in idxes}
        for i, s in zip(application_ids, scores):
            id2scores[i].append(s)
        s_w_context = [np.mean(id2scores[i]) for i in idxes]
        
        df = evaluation_df[~evaluation_df['w_context'] != True]
        application_ids = df['problem_id']
        scores = df[name + '_' + metric]
        id2scores = {i: [] for i in idxes}
        for i, s in zip(application_ids, scores):
            id2scores[i].append(s)
        s_wout_context = [np.mean(id2scores[i]) for i in idxes]
        p = ttest_rel(s_w_context, s_wout_context)[1]
        
        if p > least_sig:
            least_sig = p
    print(metric, least_sig)
print('=======================================================')
print('Calculate the correlation between each metric for each pair of evaluators')
name_pairs = [(names[i], names[j]) for i in range(3) for j in range(3) if i < j]
metric_pairs = [(metrics[i], metrics[j]) for i in range(3) for j in range(3)]
for p1, p2 in name_pairs:
    d = defaultdict(lambda: defaultdict(float))
    for metric1, metric2 in metric_pairs:
        d[p1 + metric1][p2 + metric2] = spearmanr(evaluation_df[p1 + '_' + metric1].values, evaluation_df[p2 + '_' + metric2].values)[0]
    
    df = pd.DataFrame(d)
    print(df.round(2).T.to_latex())