import io
import pickle
import re
import urllib
import time
import pandas as pd
import numpy as np
import io, os

import numpy as np
import scipy.spatial.distance as dist
from scipy.stats import norm
from scipy.optimize import minimize

from collections import defaultdict, namedtuple

from nltk.corpus import stopwords as sw
from gensim.utils import simple_preprocess
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from tqdm import trange

from util import GSD_Definition, GSD_Word

import torch
from transformers import GPT2LMHeadModel

from sentence_transformers import SentenceTransformer

# Defining Data Structs and Helper Functions
Definition = namedtuple('Definition', ['word', 'type', 'def_sent', 'ex_sents', 'stamps'])

re_hex = re.compile(r"\\x[a-f0-9][a-f0-9]")
re_spacechar = re.compile(r"\\(n|t)")
def proc_def(sent):
    return re_spacechar.sub('', re_hex.sub('', sent))

stopwords = set(sw.words('english'))

# Region Classifier Function
def classify_region(def_entry):
    #############
    # 0 - US
    # 1 - UK
    # 2 - Shared
    #############
            
    stamp_set = set([s[1] for s in def_entry.stamps])
    if '[US]' in stamp_set and '[UK]' in stamp_set:
        return 2
    elif '[US]' in stamp_set:
        return 0
    else:
        return 1
    
def tag2str(tag):
    if tag==0:
        return '[US]'
    if tag==1:
        return '[UK]'
    if tag==2:
        return '[Shared]'

def tags2str(tags):
    results = []
    for tag in tags:
        results.append(tag2str(tag))
    return results

def normalize_pair(a, b, ep=0):
    tmp = a+b+ep*2
    return ((a+ep)/tmp, (b+ep)/tmp)

def normalize_L2(array, axis=1):
    if axis == 1:
        return array / np.linalg.norm(array, axis=1)[:, np.newaxis]
    if axis == 0:
        return array / np.linalg.norm(array, axis=0)[np.newaxis, :]

def SBERT_encode(model, sentences):
    sbert_embeddings = np.asarray(model.encode(sentences))
    return normalize_L2(sbert_embeddings, axis=1)

data_GSD_raw = np.load('GSD_sample_data.npy', allow_pickle=True)

GSD_by_word = defaultdict(list)

for i in trange(data_GSD_raw.shape[0]):
    entry = data_GSD_raw[i]
    for d in entry.definitions:
        def_sent_proc = proc_def(d.def_sent)
        def_entry = Definition(entry.word, entry.pos, def_sent_proc, d.contexts, d.stamps)
        if def_entry.stamps[0][0] < 1800:
            continue
        GSD_by_word[entry.word].append(def_entry)

MIN_REGION = 5

GSD_entries = defaultdict(list)
GSD_regions = defaultdict(list)
GSD_dates = defaultdict(list)

GSD_entries_shared = defaultdict(list)

GSD_entries = defaultdict(list)
GSD_regions = defaultdict(list)
GSD_dates = defaultdict(list)

GSD_entries_shared = defaultdict(list)

for word in GSD_by_word.keys():
    entries_all = GSD_by_word[word]
    regions_all = []
    
    for entry in entries_all:
        region = classify_region(entry)
        regions_all.append(region)
    regions_all = np.asarray(regions_all)
    
    entries = [entries_all[i] for i in range(len(entries_all)) if regions_all[i] != 2]
    regions = regions_all[regions_all != 2]
    
    date_ind = np.argsort([entry.stamps[0][0] for entry in entries])
    entries_sorted = [entries[i] for i in date_ind]
    regions_sorted = regions[date_ind]
    
    dates = np.asarray([entry.stamps[0][0] for entry in entries_sorted])
    if np.sum(dates < 2000) == 0:
        continue
    if np.sum(dates >= 2000) == 0:
        continue
    
    if np.min([np.sum(regions==i) for i in range(2)]) >= MIN_REGION:
        GSD_entries[word] = entries_sorted
        GSD_regions[word] = regions_sorted
        GSD_dates[word] = dates
        
        GSD_entries_shared[word] = [entries_all[i] for i in range(len(entries_all)) if regions_all[i] == 2]
        
exp_words = list(GSD_entries.keys())
print(exp_words)

ngrams_cache = pickle.load(open('ngrams_cache.pickle', 'rb'))

def create_url(word, yr_start, yr_end, corpus='[US]', case_insensitive=True):
    url = "https://books.google.com/ngrams/graph?content="
    url += word
    if case_insensitive == True:
        url += "&case_insensitive=on"
    url += '&year_start='
    url += str(yr_start)
    url += '&year_end='
    url += str(yr_end)
    url += '&corpus='
    if corpus == '[US]':
        url += str(28)
    if corpus == '[UK]':
        url += str(29)
    url += '&smoothing=0'
    return url

def is_ascii(word):
    """Check if a word contains only ASCII characters."""
    return all(ord(c) < 128 for c in word)

def url_query(word, yr_start, yr_end, corpus='[US]'):
    # Filter out words that contain non-ASCII characters
    if not is_ascii(word):
        print(f"Skipping word with non-ASCII character: {word}")
        return [0.]*10  # Skip the request if the word is not ASCII
    
    encoded_word = urllib.parse.quote(word)
    url = create_url(encoded_word, yr_start, yr_end, corpus)
    try:
        r = urllib.request.urlopen(url)
        for line in str(r.read()).split('\\n'):
            if 'ngrams.data = ' in line:
                if 'JSON.parse' in line:
                    results = [0.]*10
                    return results
                else:
                    if 'ngram' in line.strip().split(':')[4][-6:]:
                        results = [float(s.strip()) for s in line.strip().split(':')[4].strip()[1:-12].split(',')]
                    else:
                        results = [float(s.strip()) for s in line.strip().split(':')[4].strip()[1:-4].split(',')]
                    break
    except urllib.error.HTTPError as err:
        if '429' in str(err):
            print("HTTP 429 Error")
            return None
        results = [0.]*10
    time.sleep(2)
    return results

def ngram_lookup(word, year, corpus='[US]'):
    if (word, year, corpus) not in ngrams_cache:
        results = url_query(word, year-10, year-1, corpus)
        while results is None:
            print('Access Blocked - Waiting on Word:', word)
            time.sleep(60)
            results = url_query(word, year-10, year-1, corpus)
        ngrams_cache[(word, year, corpus)] = np.mean(results)
    return ngrams_cache[(word, year, corpus)]

# Running Models on the Data
N_trials = 100
MEM = 30000
categories = ['[US]', '[UK]']
model_tags = ['sense_freq', 'sense_freq_shared', \
              '1nn', 'knn', 'tree', 'prototype', 'exemplar', 'exemplar_opt', \
              '1nn_shared', 'knn_shared', 'tree_shared', 'prototype_shared', 'exemplar_shared', 'exemplar_opt_shared']

embedder = SentenceTransformer('bert-base-nli-mean-tokens')

correct_counts_sample = {(category, n):defaultdict(int) for category in categories for n in range(N_trials)}
trivial_counts_sample = {(category, n):0 for category in categories for n in range(N_trials)}
pred_count_sample = {(category, n):0 for category in categories for n in range(N_trials)}

for n in range(N_trials):
    print("[Trial %d]" % (n+1))
    time.sleep(0.5)
    for t in trange(len(exp_words)):
        word = exp_words[t]

        entries = GSD_entries[word]
        regions = GSD_regions[word]
        dates = GSD_dates[word]

        def_sents = [entry.def_sent for entry in entries]
        def_embeds = SBERT_encode(embedder, def_sents)
        embed_dists = -1 * np.square(dist.cdist(def_embeds, def_embeds))

        context_sents = []
        for entry in entries:
            if len(entry.ex_sents) > 0:
                c = 0
                while entry.stamps[c] not in entry.ex_sents:
                    c += 1
                context_sents.append(entry.ex_sents[entry.stamps[c]])
            else:
                context_sents.append(None)

        entries_shared = GSD_entries_shared[word]

        if len(entries_shared) > 0:
            def_sents_shared = [entry.def_sent for entry in entries_shared]
            def_embeds_shared = SBERT_encode(embedder, def_sents_shared)
            embed_dists_shared = -1 * np.square(dist.cdist(def_embeds, def_embeds_shared))

            us_shared_inds = []
            uk_shared_inds = []

            for i in range(len(entries)):

                date = dates[i]

                us_shared_pos = set()
                uk_shared_pos = set()
                for j in range(len(entries_shared)):
                    seen_us = False
                    seen_uk = False
                    for stamp in entries_shared[j].stamps:
                        if stamp[0] < date:
                            if stamp[1] == '[US]' and not seen_us:
                                seen_us = True
                                if stamp[0] >= date-MEM:
                                    us_shared_pos.add(j)
                            if stamp[1] == '[UK]' and not seen_uk:
                                seen_uk = True
                                if stamp[0] >= date-MEM:
                                    uk_shared_pos.add(j)
                us_shared_inds.append(np.asarray(list(us_shared_pos)))
                uk_shared_inds.append(np.asarray(list(uk_shared_pos)))
                    
        chain_memstart = []
        for i in range(len(dates)):
            memstart = 0
            while memstart < i:
                if dates[memstart] >= dates[i]-MEM:
                    break
                memstart += 1
            chain_memstart.append(memstart)

        exemplar_valid_pos = []
        for i in range(len(entries)):
            observed = np.asarray([False, False])
            for j in range(chain_memstart[i], i):
                observed[regions[j]] = True
            if np.all(observed):
                exemplar_valid_pos.append(i)
        exemplar_valid_pos = np.asarray(exemplar_valid_pos, dtype=np.int32)

        h_old = defaultdict(lambda:1)

        priors = {'semantic_freq':[(0.5,0.5)], 'semantic_major':[(0.5,0.5)], \
                  'context_freq':[(0.5,0.5)], 'context_major':[(0.5,0.5)], \
                  'form_need':[(0.5,0.5)]}

        for p in range(1, len(dates)):

            example_regions = regions[:p]

            if dates[p] < 2000:
                for key in priors.keys():
                    priors[key].append((0.5,0.5))
                continue

            def_sent = def_sents[p]
            date = dates[p]
            
            # Form Need

            us_slang_freq = ngram_lookup(word, date, '[US]')
            uk_slang_freq = ngram_lookup(word, date, '[UK]')

            priors['form_need'].append(normalize_pair(us_slang_freq, uk_slang_freq, ep=1e-8))

            # Semantic Need

            content_words = [w for w in simple_preprocess(def_sent) if w not in stopwords]

            us_freq_total = 0
            uk_freq_total = 0

            us_more_freq = 0
            uk_more_freq = 0

            for content_word in content_words:
                us_freq = ngram_lookup(content_word, date, '[US]')
                uk_freq = ngram_lookup(content_word, date, '[UK]')

                us_freq_total += us_freq
                uk_freq_total += uk_freq

                if uk_freq > us_freq:
                    uk_more_freq += 1
                else:
                    us_more_freq += 1

            priors['semantic_freq'].append(normalize_pair(us_freq_total, uk_freq_total, ep=1e-8))
            priors['semantic_major'].append(normalize_pair(us_more_freq, uk_more_freq, ep=1))

            # Context Need

            if context_sents[p] is None:
                priors['context_freq'].append((0.5, 0.5))
                priors['context_major'].append((0.5, 0.5))
            else:

                context_sent = context_sents[p]
                date = dates[p]

                content_words = [w for w in simple_preprocess(context_sent) if (w not in stopwords and w != word)]

                us_freq_total = 0
                uk_freq_total = 0

                us_more_freq = 0
                uk_more_freq = 0

                for content_word in content_words:
                    us_freq = ngram_lookup(content_word, date, '[US]')
                    uk_freq = ngram_lookup(content_word, date, '[UK]')

                    us_freq_total += us_freq
                    uk_freq_total += uk_freq

                    if uk_freq > us_freq:
                        uk_more_freq += 1
                    else:
                        us_more_freq += 1

                priors['context_freq'].append(normalize_pair(us_freq_total, uk_freq_total, ep=1e-8))
                priors['context_major'].append(normalize_pair(us_more_freq, uk_more_freq, ep=1))

        # Sample test senses
        
        chain_start = 1
        while dates[chain_start] < 2000:
            chain_start += 1

        chain_us = np.arange(chain_start, len(dates))[regions[chain_start:]==0]
        chain_uk = np.arange(chain_start, len(dates))[regions[chain_start:]==1]

        N_sample = min(len(chain_us), len(chain_uk))
        if N_sample == 0:
            continue

        if len(chain_us) > N_sample:
            chain_us = chain_us[np.random.choice(len(chain_us), N_sample, replace=False)]
        if len(chain_uk) > N_sample:
            chain_uk = chain_uk[np.random.choice(len(chain_uk), N_sample, replace=False)]

        chain = np.sort(np.concatenate((chain_us, chain_uk)))
            
        for chain_pos in chain:

            if dates[chain_pos] < 2000:
                continue

            example_regions = regions[chain_memstart[chain_pos]:chain_pos]
            target_region = regions[chain_pos]
            target_str = tag2str(target_region)

            preds = defaultdict(int)
            pred_count_sample[(target_str, n)] += 1

            # Communicative Need Models

            preds['form_need'] = priors['form_need'][chain_pos][0] < priors['form_need'][chain_pos][1]

            preds['semantic_freq'] = priors['semantic_freq'][chain_pos][0] < priors['semantic_freq'][chain_pos][1]
            preds['semantic_major'] = priors['semantic_major'][chain_pos][0] < priors['semantic_major'][chain_pos][1]

            preds['context_freq'] = priors['context_freq'][chain_pos][0] < priors['context_freq'][chain_pos][1]
            preds['context_major'] = priors['context_major'][chain_pos][0] < priors['context_major'][chain_pos][1]

            skip_trivial = False

            if np.sum(example_regions==0) == 0:
                for m_tag in model_tags:
                    preds[m_tag] = 1
                skip_trivial = True

            if np.sum(example_regions==1) == 0:
                for m_tag in model_tags:
                    preds[m_tag] = 0
                skip_trivial = True

            if skip_trivial:
                for key, value in preds.items():
                    if value == target_region:
                        correct_counts_sample[(target_str, n)][key] += 1
                continue

            # Sense Frequency

            preds['sense_freq'] = np.sum(example_regions==0) < np.sum(example_regions==1)
            if len(entries_shared) > 0:
                preds['sense_freq_shared'] = (np.sum(example_regions==0) + len(us_shared_inds[chain_pos])) < \
                                             (np.sum(example_regions==1) + len(uk_shared_inds[chain_pos]))
            else:
                preds['sense_freq_shared'] = preds['sense_freq']

            # Semantic Chaining

            us_dists = np.exp(embed_dists[chain_pos][chain_memstart[chain_pos]:chain_pos][example_regions==0])
            uk_dists = np.exp(embed_dists[chain_pos][chain_memstart[chain_pos]:chain_pos][example_regions==1])

            us_def_embeds = def_embeds[chain_memstart[chain_pos]:chain_pos][example_regions==0]
            uk_def_embeds = def_embeds[chain_memstart[chain_pos]:chain_pos][example_regions==1]

            us_prototype = np.mean(us_def_embeds, axis=0)
            uk_prototype = np.mean(uk_def_embeds, axis=0)

            us_proto_dist = np.linalg.norm(us_prototype-def_embeds[chain_pos])
            uk_proto_dist = np.linalg.norm(uk_prototype-def_embeds[chain_pos])

            # KNN Prediction
            num_trainers = len(def_embeds[chain_memstart[chain_pos]:chain_pos])
            K = min(5, num_trainers)  # Number of neighbors
            train_embeds = def_embeds[chain_memstart[chain_pos]:chain_pos]  # Training embeddings
            train_labels = example_regions  # 0 for US, 1 for UK

            # Decision Tree 
            tree = DecisionTreeClassifier()
            tree.fit(train_embeds, train_labels)
            preds['tree'] = tree.predict(def_embeds[chain_pos][np.newaxis, :])[0]
            
            if len(entries_shared) > 0:
            # Train embeds with shared senses
                train_embeds_shared = np.concatenate((us_def_embeds, uk_def_embeds), axis=0)
                train_labels_shared = np.concatenate((np.zeros(len(us_def_embeds)), np.ones(len(uk_def_embeds))))
                tree_shared = DecisionTreeClassifier()
                tree_shared.fit(train_embeds_shared, train_labels_shared)
                preds['tree_shared'] = tree_shared.predict(def_embeds[chain_pos][np.newaxis, :])[0]
            else:
                preds['tree_shared'] = preds['tree']

            # Fit KNN model and predict
            knn = KNeighborsClassifier(n_neighbors=K)
            knn.fit(train_embeds, train_labels)

            preds['1nn'] = int(np.max(us_dists) < np.max(uk_dists))
            preds['knn'] = knn.predict(def_embeds[chain_pos][np.newaxis, :])[0]
            preds['exemplar'] = int(np.mean(us_dists) < np.mean(uk_dists))
            preds['prototype'] = int(us_proto_dist > uk_proto_dist)

            # Optimize kernel width parameter (h) if training data is available
            # Only need to optimize exemplar since we're not using a prior

            exemplar_starts = exemplar_valid_pos[exemplar_valid_pos < chain_pos]
            if exemplar_starts.shape[0] != 0:

                def compute_exemplar_nll(h=1):
                    nll = 0

                    for pred_pos in exemplar_starts:
                            
                        region_sub = regions[chain_memstart[pred_pos]:pred_pos]

                        us_dists = np.exp(embed_dists[pred_pos][chain_memstart[pred_pos]:pred_pos][region_sub==0] / h)
                        uk_dists = np.exp(embed_dists[pred_pos][chain_memstart[pred_pos]:pred_pos][region_sub==1] / h)

                        pred_dist = np.asarray([np.mean(us_dists), np.mean(uk_dists)])
                        pred_dist = pred_dist / np.sum(pred_dist)
                        nll += np.log(pred_dist[regions[pred_pos]])
                    return -1 * nll

                results = minimize(compute_exemplar_nll, [h_old['exemplar']], bounds=((10**-2, 10**2),))
                h_old['exemplar'] = results.x[0]

                us_dists = np.exp(embed_dists[chain_pos][chain_memstart[chain_pos]:chain_pos][example_regions==0] / h_old['exemplar_opt'])
                uk_dists = np.exp(embed_dists[chain_pos][chain_memstart[chain_pos]:chain_pos][example_regions==1] / h_old['exemplar_opt'])

                preds['exemplar'] = int(np.mean(us_dists) < np.mean(uk_dists))

                
            # Chaining with shared senses

            if len(entries_shared) == 0:
                preds['1nn_shared'] = preds['1nn']
                preds['knn_shared'] = preds['knn']
                preds['exemplar_shared'] = preds['exemplar']
                preds['prototype_shared'] = preds['prototype']
            else:
                us_shared_pos = us_shared_inds[chain_pos]
                uk_shared_pos = uk_shared_inds[chain_pos]

                if us_shared_pos.shape[0] > 0:
                    us_dists = np.exp(np.concatenate((embed_dists[chain_pos][chain_memstart[chain_pos]:chain_pos][example_regions==0], embed_dists_shared[chain_pos][us_shared_pos])))
                    us_def_embeds = np.concatenate((def_embeds[chain_memstart[chain_pos]:chain_pos][example_regions==0], def_embeds_shared[us_shared_pos]), axis=0)
                else:
                    us_dists = np.exp(embed_dists[chain_pos][chain_memstart[chain_pos]:chain_pos][example_regions==0])
                    us_def_embeds = def_embeds[chain_memstart[chain_pos]:chain_pos][example_regions==0]

                if uk_shared_pos.shape[0] > 0:
                    uk_dists = np.exp(np.concatenate((embed_dists[chain_pos][chain_memstart[chain_pos]:chain_pos][example_regions==1], embed_dists_shared[chain_pos][uk_shared_pos])))
                    uk_def_embeds = np.concatenate((def_embeds[chain_memstart[chain_pos]:chain_pos][example_regions==1], def_embeds_shared[uk_shared_pos]), axis=0)
                else:
                    uk_dists = np.exp(embed_dists[chain_pos][chain_memstart[chain_pos]:chain_pos][example_regions==1])
                    uk_def_embeds = def_embeds[chain_memstart[chain_pos]:chain_pos][example_regions==1]

                us_prototype = np.mean(us_def_embeds, axis=0)
                uk_prototype = np.mean(uk_def_embeds, axis=0)

                us_proto_dist = np.linalg.norm(us_prototype-def_embeds[chain_pos])
                uk_proto_dist = np.linalg.norm(uk_prototype-def_embeds[chain_pos])

                # KNN Prediction with shared senses
                train_embeds_shared = np.concatenate((us_def_embeds, uk_def_embeds), axis=0)
                train_labels_shared = np.concatenate((np.zeros(len(us_def_embeds)), np.ones(len(uk_def_embeds))))

                num_trainers_shared = len(train_embeds_shared)
                K = min(5, num_trainers_shared)  # Number of neighbors
                knn_shared = KNeighborsClassifier(n_neighbors=K)
                knn_shared.fit(train_embeds_shared, train_labels_shared)

                preds['1nn_shared'] = int(np.max(us_dists) < np.max(uk_dists))
                preds['knn_shared'] = knn_shared.predict(def_embeds[chain_pos][np.newaxis, :])[0]
                preds['exemplar_shared'] = int(np.mean(us_dists) < np.mean(uk_dists))
                preds['prototype_shared'] = int(us_proto_dist > uk_proto_dist)

                exemplar_starts = exemplar_valid_pos[exemplar_valid_pos < chain_pos]
                if exemplar_starts.shape[0] != 0:

                    def compute_exemplar_nll(h=1):
                        nll = 0

                        for pred_pos in exemplar_starts:
                            region_sub = regions[chain_memstart[pred_pos]:pred_pos]

                            us_shared_pos = us_shared_inds[pred_pos]
                            uk_shared_pos = uk_shared_inds[pred_pos]

                            if us_shared_pos.shape[0] > 0:
                                us_dists = np.exp(np.concatenate((embed_dists[pred_pos][chain_memstart[pred_pos]:pred_pos][region_sub==0], embed_dists_shared[pred_pos][us_shared_pos])) / h)
                            else:
                                us_dists = np.exp(embed_dists[pred_pos][chain_memstart[pred_pos]:pred_pos][region_sub==0] / h)

                            if uk_shared_pos.shape[0] > 0:
                                uk_dists = np.exp(np.concatenate((embed_dists[pred_pos][chain_memstart[pred_pos]:pred_pos][region_sub==1], embed_dists_shared[pred_pos][uk_shared_pos])) / h)
                            else:
                                uk_dists = np.exp(embed_dists[pred_pos][chain_memstart[pred_pos]:pred_pos][region_sub==1] / h)

                            pred_dist = np.asarray([np.mean(us_dists), np.mean(uk_dists)])
                            pred_dist = pred_dist / np.sum(pred_dist)
                            nll += np.log(pred_dist[regions[pred_pos]])
                        return -1 * nll

                    results = minimize(compute_exemplar_nll, [h_old['exemplar_shared']], bounds=((10**-2, 10**2),))
                    h_old['exemplar_shared'] = results.x[0]

                    us_shared_pos = us_shared_inds[chain_pos]
                    uk_shared_pos = uk_shared_inds[chain_pos]

                    if us_shared_pos.shape[0] > 0:
                        us_dists = np.exp(np.concatenate((embed_dists[chain_pos][chain_memstart[chain_pos]:chain_pos][example_regions==0], embed_dists_shared[chain_pos][us_shared_pos])) / h_old['exemplar_opt_shared'])
                    else:
                        us_dists = np.exp(embed_dists[chain_pos][chain_memstart[chain_pos]:chain_pos][example_regions==0] / h_old['exemplar_opt_shared'])

                    if uk_shared_pos.shape[0] > 0:
                        uk_dists = np.exp(np.concatenate((embed_dists[chain_pos][chain_memstart[chain_pos]:chain_pos][example_regions==1], embed_dists_shared[chain_pos][uk_shared_pos])) / h_old['exemplar_opt_shared'])
                    else:
                        uk_dists = np.exp(embed_dists[chain_pos][chain_memstart[chain_pos]:chain_pos][example_regions==1] / h_old['exemplar_opt_shared'])

                    preds['exemplar_shared'] = int(np.mean(us_dists) < np.mean(uk_dists))

               
            # Collect Results
            for key, value in preds.items():
                if value == target_region:
                    correct_counts_sample[(target_str, n)][key] += 1

model_list = {'Baseline': ['sense_freq', 'sense_freq_shared'],\
                      'Need':['form_need', 'semantic_freq', 'semantic_major', 'context_freq', 'context_major'],\
                      'Chaining':['1nn', 'knn', 'tree', 'prototype', 'exemplar'],\
                      'Chaining - Shared':['1nn_shared', 'tree_shared', 'knn_shared', 'prototype_shared', 'exemplar_shared']}

model_list = {'Baseline': ['sense_freq', 'sense_freq_shared'],\
                      'Need':['form_need', 'semantic_freq', 'semantic_major', 'context_freq', 'context_major'],\
                      'Chaining':['1nn', 'knn', 'tree', 'prototype', 'exemplar'],\
                      'Chaining - Shared':['1nn_shared', 'tree_shared', 'knn_shared', 'prototype_shared', 'exemplar_shared']}

# Redirect print output to a StringIO object
output_buffer = io.StringIO()

print("%51s%12s%14s" % ("[US]", "[UK]", "Total"), file=output_buffer)
for group, models in model_list.items():
    print("["+group.upper()+"]", file=output_buffer)
    for model in models:
        us_correct = np.asarray([correct_counts_sample[('[US]', n)][model] for n in range(N_trials)])
        us_total = np.asarray([pred_count_sample[('[US]', n)] for n in range(N_trials)])

        uk_correct = np.asarray([correct_counts_sample[('[UK]', n)][model] for n in range(N_trials)])
        uk_total = np.asarray([pred_count_sample[('[UK]', n)] for n in range(N_trials)])

        print("%35s:   %.1f (%.2f)  %.1f (%.2f)  %.1f (%.2f)" % \
              (model.upper(), \
               np.mean(us_correct / us_total * 100), np.std(us_correct / us_total * 100),\
               np.mean(uk_correct / uk_total * 100), np.std(uk_correct / uk_total * 100),\
               np.mean((us_correct + uk_correct) / (us_total + uk_total) * 100), np.std((us_correct + uk_correct) / (us_total + uk_total) * 100)), file=output_buffer)
    print("", file=output_buffer)

# Parse the output from the buffer
output_lines = output_buffer.getvalue().strip().split("\n")
data = []

for line in output_lines:
    line = line.strip()
    if line.startswith("[") or line.startswith("%") or not line:
        data.append([line])
    else:
        parts = line.split(":", 1)
        if len(parts) == 2:
            model = parts[0].strip()
            stats = parts[1].split()
            if len(stats) >= 6:
                us_mean, us_std = stats[0], stats[1].strip("()")
                uk_mean, uk_std = stats[2], stats[3].strip("()")
                total_mean, total_std = stats[4], stats[5].strip("()")
                data.append([model, us_mean, us_std, uk_mean, uk_std, total_mean, total_std])

# Convert to DataFrame
columns = ["Model/Group", "US Mean", "US Std", "UK Mean", "UK Std", "Total Mean", "Total Std"]
df = pd.DataFrame(data, columns=columns)

# Specify Excel output file
output_file = "output_results.xlsx"

# Create a new Excel file if it doesn't exist
if not os.path.exists(output_file):
    pd.DataFrame().to_excel(output_file, engine='openpyxl', index=False)

# Append to Excel file
with pd.ExcelWriter(output_file, engine='openpyxl', mode='a', if_sheet_exists='new') as writer:
    df.to_excel(writer, sheet_name='Results', index=False)

print("Results have been appended to the Excel file:", output_file)
