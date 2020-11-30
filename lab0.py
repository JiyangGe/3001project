#!/usr/bin/env python
'''A basic script to test our environment'''

import os
import pandas as pd
import lightfm
import lenskit
import numpy as np
from statistics import mean

from lenskit.datasets import MovieLens
from lenskit.crossfold import partition_users, SampleFrac

from lenskit import batch, topn, datasets, util
from lenskit.metrics.predict import rmse, mae
from lenskit.crossfold import partition_users, SampleN 
from lenskit.algorithms import als, Recommender, basic
# from lenskit.eval.traintest.predict.CoveragePredictMetric import Coverage

from sklearn.metrics import roc_auc_score
import numpy as np

import pandas as pd
 
if __name__ == '__main__':


    train = pd.read_csv('data/beer/train.csv')
    validation = pd.read_csv('data/beer/validation.csv')
    test = pd.read_csv('data/beer/test.csv')
    # train = pd.read_csv('data/jester/clean/train.csv')
    # validation = pd.read_csv('data/jester/clean/validation.csv')
    # test = pd.read_csv('data/jester/clean/test.csv')

    algo = basic.Bias()
    # algo = als.BiasedMF()
    
    train = train[['reviewer_id', 'beer_beerid', 'review_overall']]
    validation=validation[['reviewer_id', 'beer_beerid', 'review_overall']]
    test =test[['reviewer_id', 'beer_beerid', 'review_overall']]
    train=train.rename(columns={'review_overall': 'rating', 'reviewer_id':'user','beer_beerid':'item'})
    validation=validation.rename(columns={'review_overall': 'rating', 'reviewer_id':'user','beer_beerid':'item'})
    test=test.rename(columns={'review_overall': 'rating', 'reviewer_id':'user','beer_beerid':'item'})

    # train = train[['user', 'item', 'rating']]
    # validation = validation[['user', 'item', 'rating']]
    # test = test[['user', 'item', 'rating']]


    def eval(aname, algo, train, test, n):
        fittable = util.clone(algo)
        fittable = Recommender.adapt(fittable)
        fittable.fit(train)

        # predict ratings
        ratings_est = fittable.predict(test[['user', 'item']])
        print(len(ratings_est))
        print(len(test['rating']))
        # now we run the recommender
        users = test.user.unique()
        recs = batch.recommend(fittable, users, n)
        # add the algorithm name for analyzability
        recs['Algorithm'] = aname

        y_true=[]
        for i in range(len(recs)):
            row = recs.iloc[i]
            user_id = row['user']
            item_id = row['item']
            boolen_ls = (test['user'] == user_id)
            chosen_rows = [i for i, x in enumerate(boolen_ls) if x]
            focs_test = test.iloc[chosen_rows]
            focs_test=focs_test[focs_test['rating']>=4]
            if (item_id in focs_test['item']):
                y_true.append('1')
            else:
                y_true.append('0')

        def coverage(preds, items, num_items):
            rec_item = []
            for i in range(len(preds)):
                # for beer
                if preds[i] >= 4:
                # for jester 
                # if preds[i] > 0:
                    rec_item.append(items[i])
            return len(set(rec_item)) / num_items

        def hit_rate(preds, labels, users, topk=10):
            user_pred_dict = {}
            hit_rates = []
            for i in range(len(preds)):
                if users[i] not in user_pred_dict:
                    user_pred_dict[users[i]] = []
                    user_pred_dict[users[i]].append((preds[i], labels[i]))
                for user in user_pred_dict:
                    user_res = sorted(user_pred_dict[user], key=lambda x: x[0])[-topk:]
                    hit_rates.append(np.sum([int(x[1]) > 0 for x in user_res])/topk)
            return np.mean(hit_rates)

        all_df = pd.concat([train,test])
        
        item_count = len(all_df['item'].unique())

        auc_score = roc_auc_score(y_true, recs['score'])
        cov = coverage(recs['score'], recs['item'], item_count)
        hit = hit_rate(recs['score'], recs['item'], recs['user'])

        return score, cov, hit
  
    # try 100 features
    auc_score, coverage, hit = eval('100', algo, train, validation, 100)
    print('auc_score for validation:',auc_score)
    print('coverage for validation:', coverage)
    print('hit for validation:', hit)

    train_val = pd.concat([train,validation])
    auc_score_test, coverage_test, hit_test = eval('100', algo, train_val, test, 100)
    print('auc_score for test:',auc_score)
    print('coverage for test:', coverage)
    print('hit for test:', hit)


