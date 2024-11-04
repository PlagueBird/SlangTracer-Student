import io
import pickle
import re
import urllib
import time
import nltk

nltk.download('stopwords')

import numpy as np
import scipy.spatial.distance as dist
from scipy.stats import norm
from scipy.optimize import minimize

from collections import defaultdict, namedtuple

from nltk.corpus import stopwords as sw
from gensim.utils import simple_preprocess
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression

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
    # 2 - AUS
    # 3 - Shared
    #############
            
    stamp_set = set([s[1] for s in def_entry.stamps])
    if '[US]' in stamp_set and '[UK]' in stamp_set and '[AUS]' in stamp_set:
        return 3
    elif '[US]' in stamp_set:
        return 0
    elif '[AUS]' in stamp_set:
        return 2
    else:
        return 1
    
def tag2str(tag):
    if tag==0:
        return '[US]'
    if tag==1:
        return '[UK]'
    if tag==2:
        return '[AUS]'
    if tag==3:
        return '[Shared]'

def tags2str(tags):
    results = []
    for tag in tags:
        results.append(tag2str(tag))
    return results

def normalize_set(a, b, c, ep=0):
    tmp = a+b+c+ep*3
    return ((a + ep) / tmp, (b + ep) / tmp, (c + ep) / tmp)

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
    if np.sum(dates < 1900) == 0:
        continue
    if np.sum(dates >= 1900) == 0:
        continue
    
    if np.min([np.sum(regions==i) for i in range(2)]) >= MIN_REGION:
        GSD_entries[word] = entries_sorted
        GSD_regions[word] = regions_sorted
        GSD_dates[word] = dates
        
        GSD_entries_shared[word] = [entries_all[i] for i in range(len(entries_all)) if regions_all[i] == 2]
        
exp_words = list(GSD_entries.keys())

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
    if corpus == '[AUS]':
        url += str(30)
    url += '&smoothing=0'
    return url

def url_query(word, yr_start, yr_end, corpus='[US]'):
    url = create_url(word, yr_start, yr_end, corpus)
    try:
        r = urllib.request.urlopen(url)
        for line in str(r.read()).split('\\n'):
            if 'ngrams.data = ' in line:
                if 'ngram' in line.strip().split(':')[4][-6:]:
                    results = [float(s.strip()) for s in line.strip().split(':')[4].strip()[1:-12].split(',')]
                else:
                    results = [float(s.strip()) for s in line.strip().split(':')[4].strip()[1:-4].split(',')]
                break
    except urllib.error.HTTPError as err:
        if '429' in str(err):
            return None
        results = [0.]*10
    time.sleep(2)
    return results

def ngram_lookup(word, year, corpus='[US]'):
    if (word, year, corpus) not in ngrams_cache:
        results = url_query(word, year-10, year-1, corpus)
        while results is None:
            print('Access Blocked - Waiting')
            time.sleep(300)
            results = url_query(word, year-10, year-1, corpus)
        ngrams_cache[(word, year, corpus)] = np.mean(results)
    return ngrams_cache[(word, year, corpus)]

# Running Models on the Data
N_trials = 20
MEM = 30000
categories = ['[US]', '[UK]', '[AUS]']
model_tags = ['sense_freq', 'sense_freq_shared', \
              'lda', 'lda_shared', 'logistic_reg', 'logistic_reg_shared', \
              '1nn', 'prototype', 'exemplar', 'exemplar_opt', \
              '1nn_shared', 'prototype_shared', 'exemplar_shared', 'exemplar_opt_shared']

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
            aus_shared_inds = []

            for i in range(len(entries)):

                date = dates[i]

                us_shared_pos = set()
                uk_shared_pos = set()
                aus_shared_pos = set()
                for j in range(len(entries_shared)):
                    seen_us = False
                    seen_uk = False
                    seen_aus = False
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
                            if stamp[1] == '[AUS]' and not seen_aus:
                                seen_aus = True
                                if stamp[0] >= date-MEM:
                                    aus_shared_pos.add(j)
                us_shared_inds.append(np.asarray(list(us_shared_pos)))
                uk_shared_inds.append(np.asarray(list(uk_shared_pos)))
                aus_shared_inds.append(np.asarray(list(aus_shared_pos)))
                    
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

            if dates[p] < 1900:
                for key in priors.keys():
                    priors[key].append((0.5,0.5))
                continue

            def_sent = def_sents[p]
            date = dates[p]
            
            # Form Need

            us_slang_freq = ngram_lookup(word, date, '[US]')
            uk_slang_freq = ngram_lookup(word, date, '[UK]')
            aus_slang_freq = ngram_lookup(word, date, '[AUS]')

            priors['form_need'].append(normalize_set(us_slang_freq, uk_slang_freq, aus_slang_freq, ep=1e-8))

            # Semantic Need

            content_words = [w for w in simple_preprocess(def_sent) if w not in stopwords]

            us_freq_total = 0
            uk_freq_total = 0
            aus_freq_total = 0

            us_more_freq = 0
            uk_more_freq = 0
            aus_more_freq = 0

            for content_word in content_words:
                us_freq = ngram_lookup(content_word, date, '[US]')
                uk_freq = ngram_lookup(content_word, date, '[UK]')
                aus_freq = ngram_lookup(content_word, date, '[AUS]')

                us_freq_total += us_freq
                uk_freq_total += uk_freq
                aus_freq_total += aus_freq

                if uk_freq > us_freq and uk_freq > aus_freq:
                    uk_more_freq += 1
                elif aus_freq > uk_freq and aus_freq > us_freq:
                    aus_more_freq += 1
                else:
                    us_more_freq += 1

            priors['semantic_freq'].append(normalize_set(us_freq_total, uk_freq_total, aus_freq_total, ep=1e-8))
            priors['semantic_major'].append(normalize_set(us_more_freq, uk_more_freq, aus_more_freq, ep=1))

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
                aus_freq_total = 0

                us_more_freq = 0
                uk_more_freq = 0
                aus_more_freq = 0

                for content_word in content_words:
                    us_freq = ngram_lookup(content_word, date, '[US]')
                    uk_freq = ngram_lookup(content_word, date, '[UK]')
                    aus_freq = ngram_lookup(content_word, date, '[AUS]')

                    us_freq_total += us_freq
                    uk_freq_total += uk_freq
                    aus_freq_total += aus_freq

                    if uk_freq > us_freq and uk_freq > aus_freq:
                        uk_more_freq += 1
                    elif aus_freq > us_freq and aus_freq > uk_freq:
                        aus_more_freq += 1
                    else:
                        us_more_freq += 1

                priors['context_freq'].append(normalize_set(us_freq_total, uk_freq_total, aus_freq_total, ep=1e-8))
                priors['context_major'].append(normalize_set(us_more_freq, uk_more_freq, aus_more_freq, ep=1))

        # Sample test senses
        
        chain_start = 1
        while dates[chain_start] < 1900:
            chain_start += 1

        chain_us = np.arange(chain_start, len(dates))[regions[chain_start:]==0]
        chain_uk = np.arange(chain_start, len(dates))[regions[chain_start:]==1]
        chain_aus = np.arange(chain_start, len(dates))[regions[chain_start:]==2]

        N_sample = min(len(chain_us), len(chain_uk), len(chain_aus))
        if N_sample == 0:
            continue

        if len(chain_us) > N_sample:
            chain_us = chain_us[np.random.choice(len(chain_us), N_sample, replace=False)]
        if len(chain_uk) > N_sample:
            chain_uk = chain_uk[np.random.choice(len(chain_uk), N_sample, replace=False)]
        if len(chain_aus) > N_sample:
            chain_aus = chain_aus[np.random.choice(len(chain_aus), N_sample, replace=False)]

        chain = np.sort(np.concatenate((chain_us, chain_uk, chain_aus)))
            
        for chain_pos in chain:

            if dates[chain_pos] < 1900:
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
                                             (np.sum(example_regions==1) + len(uk_shared_inds[chain_pos])) < \
                                             (np.sum(example_regions==1) + len(aus_shared_inds[chain_pos]))
            else:
                preds['sense_freq_shared'] = preds['sense_freq']

            # LDA method

            if chain_pos - chain_memstart[chain_pos] == 2:
                # Need more examples than number of classes
                preds['lda'] = 0
                preds['lda_shared'] = 0
            else:
                lda = LinearDiscriminantAnalysis(n_components=1)
                
                lda.fit(def_embeds[chain_memstart[chain_pos]:chain_pos], example_regions)
                preds['lda'] = lda.predict(def_embeds[chain_pos][np.newaxis, :])

                if len(entries_shared) == 0:
                    preds['lda_shared'] = preds['lda']
                else:
                    us_shared_pos = [i for i in us_shared_inds[chain_pos] if i not in uk_shared_inds[chain_pos] and i not in aus_shared_inds[chain_pos]]
                    uk_shared_pos = [i for i in uk_shared_inds[chain_pos] if i not in us_shared_inds[chain_pos] and i not in aus_shared_inds[chain_pos]]
                    aus_shared_pos = [i for i in aus_shared_inds[chain_pos] if i not in us_shared_inds[chain_pos] and i not in us_shared_inds[chain_pos]]

                    embeds_tmp = np.concatenate((def_embeds[chain_memstart[chain_pos]:chain_pos], def_embeds_shared[us_shared_pos], def_embeds_shared[uk_shared_pos], def_embeds_shared[aus_shared_pos]), axis=0)
                    regions_tmp = np.concatenate((example_regions, [0]*len(us_shared_pos), [1]*len(uk_shared_pos, [2]*len(aus_shared_pos))))

                    lda_shared = LinearDiscriminantAnalysis(n_components=1)
                    lda_shared.fit(embeds_tmp, regions_tmp)
                    preds['lda_shared'] = lda_shared.predict(def_embeds[chain_pos][np.newaxis, :])

            # Logistic Regression

            lr = LogisticRegression().fit(def_embeds[chain_memstart[chain_pos]:chain_pos], example_regions)
            preds['logistic_reg'] = lr.predict(def_embeds[chain_pos][np.newaxis, :])

            if len(entries_shared) == 0:
                preds['logistic_reg_shared'] = preds['logistic_reg']
            else:
                lr_shared = LogisticRegression().fit(embeds_tmp, regions_tmp)
                preds['logistic_reg_shared'] = lr_shared.predict(def_embeds[chain_pos][np.newaxis, :])

            # Semantic Chaining

            us_dists = np.exp(embed_dists[chain_pos][chain_memstart[chain_pos]:chain_pos][example_regions==0])
            uk_dists = np.exp(embed_dists[chain_pos][chain_memstart[chain_pos]:chain_pos][example_regions==1])
            aus_dists = np.exp(embed_dists[chain_pos][chain_memstart[chain_pos]:chain_pos][example_regions==2])


            us_def_embeds = def_embeds[chain_memstart[chain_pos]:chain_pos][example_regions==0]
            uk_def_embeds = def_embeds[chain_memstart[chain_pos]:chain_pos][example_regions==1]
            aus_def_embeds = def_embeds[chain_memstart[chain_pos]:chain_pos][example_regions==2]

            us_prototype = np.mean(us_def_embeds, axis=0)
            uk_prototype = np.mean(uk_def_embeds, axis=0)
            aus_prototype = np.mean(aus_def_embeds, axis=0)

            us_proto_dist = np.linalg.norm(us_prototype-def_embeds[chain_pos])
            uk_proto_dist = np.linalg.norm(uk_prototype-def_embeds[chain_pos])
            aus_proto_dist = np.linalg.norm(aus_prototype-def_embeds[chain_pos])

            nn_int = 0
            if np.max(us_dists) < np.max(uk_dists) and np.max(us_dists) < np.max(aus_dists):
                nn_int = 0
            elif np.max(uk_dists) < np.max(us_dists) and np.max(uk_dists) < np.max(aus_dists):
                nn_int = 1
            else:
                nn_int = 2
            preds['1nn'] = int(nn_int)

            exemplar_int = 0
            if np.mean(us_dists) < np.mean(uk_dists) and np.mean(us_dists) < np.mean(aus_dists):
                exemplar_int = 0
            elif np.mean(uk_dists) < np.mean(us_dists) and np.mean(uk_dists) < np.mean(aus_dists):
                exemplar_int = 1
            else:
                exemplar_int = 2
            preds['exemplar'] = int(exemplar_int)

            proto_test = 0
            if us_proto_dist > uk_proto_dist and us_proto_dist > uk_proto_dist:
                proto_test = 0
            elif uk_proto_dist > us_proto_dist and uk_proto_dist > aus_proto_dist:
                proto_test = 1
            else:
                proto_test = 2
            preds['prototype'] = int(proto_test)

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
                        aus_dists = np.exp(embed_dists[pred_pos][chain_memstart[pred_pos]:pred_pos][region_sub==2] / h)

                        pred_dist = np.asarray([np.mean(us_dists), np.mean(uk_dists), np.mean(aus_dists)])
                        pred_dist = pred_dist / np.sum(pred_dist)
                        nll += np.log(pred_dist[regions[pred_pos]])
                    return -1 * nll

                results = minimize(compute_exemplar_nll, [h_old['exemplar']], bounds=((10**-2, 10**2),))
                h_old['exemplar'] = results.x[0]

                us_dists = np.exp(embed_dists[chain_pos][chain_memstart[chain_pos]:chain_pos][example_regions==0] / h_old['exemplar_opt'])
                uk_dists = np.exp(embed_dists[chain_pos][chain_memstart[chain_pos]:chain_pos][example_regions==1] / h_old['exemplar_opt'])
                aus_dists = np.exp(embed_dists[chain_pos][chain_memstart[chain_pos]:chain_pos][example_regions==2] / h_old['exemplar_opt'])


                preds['exemplar'] = int(np.mean(us_dists) < np.mean(uk_dists))

                
            # Chaining with shared senses

            if len(entries_shared) == 0:
                preds['1nn_shared'] = preds['1nn']
                preds['exemplar_shared'] = preds['exemplar']
                preds['prototype_shared'] = preds['prototype']
            else:
                us_shared_pos = us_shared_inds[chain_pos]
                uk_shared_pos = uk_shared_inds[chain_pos]
                us_shared_pos = aus_shared_inds[chain_pos]

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

                if aus_shared_pos.shape[0] > 0:
                    aus_dists = np.exp(np.concatenate((embed_dists[chain_pos][chain_memstart[chain_pos]:chain_pos][example_regions==0], embed_dists_shared[chain_pos][aus_shared_pos])))
                    aus_def_embeds = np.concatenate((def_embeds[chain_memstart[chain_pos]:chain_pos][example_regions==0], def_embeds_shared[aus_shared_pos]), axis=0)
                else:
                    us_dists = np.exp(embed_dists[chain_pos][chain_memstart[chain_pos]:chain_pos][example_regions==0])
                    us_def_embeds = def_embeds[chain_memstart[chain_pos]:chain_pos][example_regions==0]

                us_prototype = np.mean(us_def_embeds, axis=0)
                uk_prototype = np.mean(uk_def_embeds, axis=0)
                aus_prototype = np.mean(aus_def_embeds, axis=0)

                us_proto_dist = np.linalg.norm(us_prototype-def_embeds[chain_pos])
                uk_proto_dist = np.linalg.norm(uk_prototype-def_embeds[chain_pos])
                aus_proto_dist = np.linalg.norm(aus_prototype-def_embeds[chain_pos])


                preds['1nn_shared'] = int((np.max(us_dists) < np.max(uk_dists)) + (np.max(uk_dists) < np.max(aus_dists)))
                preds['exemplar_shared'] = int((np.mean(us_dists) < np.mean(uk_dists)) + (np.mean(uk_dists) < np.mean(aus_dists)))
                preds['prototype_shared'] = int((us_proto_dist > uk_proto_dist) + (uk_proto_dist > aus_proto_dist))

                exemplar_starts = exemplar_valid_pos[exemplar_valid_pos < chain_pos]
                if exemplar_starts.shape[0] != 0:

                    def compute_exemplar_nll(h=1):
                        nll = 0

                        for pred_pos in exemplar_starts:
                            region_sub = regions[chain_memstart[pred_pos]:pred_pos]

                            us_shared_pos = us_shared_inds[pred_pos]
                            uk_shared_pos = uk_shared_inds[pred_pos]
                            aus_shared_pos = aus_shared_inds[pred_pos]

                            if us_shared_pos.shape[0] > 0:
                                us_dists = np.exp(np.concatenate((embed_dists[pred_pos][chain_memstart[pred_pos]:pred_pos][region_sub==0], embed_dists_shared[pred_pos][us_shared_pos])) / h)
                            else:
                                us_dists = np.exp(embed_dists[pred_pos][chain_memstart[pred_pos]:pred_pos][region_sub==0] / h)

                            if uk_shared_pos.shape[0] > 0:
                                uk_dists = np.exp(np.concatenate((embed_dists[pred_pos][chain_memstart[pred_pos]:pred_pos][region_sub==1], embed_dists_shared[pred_pos][uk_shared_pos])) / h)
                            else:
                                uk_dists = np.exp(embed_dists[pred_pos][chain_memstart[pred_pos]:pred_pos][region_sub==1] / h)

                            if aus_shared_pos.shape[0] > 0:
                                aus_dists = np.exp(np.concatenate((embed_dists[pred_pos][chain_memstart[pred_pos]:pred_pos][region_sub==1], embed_dists_shared[pred_pos][uk_shared_pos])) / h)
                            else:
                                aus_dists = np.exp(embed_dists[pred_pos][chain_memstart[pred_pos]:pred_pos][region_sub==1] / h)

                            pred_dist = np.asarray([np.mean(us_dists), np.mean(uk_dists), np.mean(aus_dists)])
                            pred_dist = pred_dist / np.sum(pred_dist)
                            nll += np.log(pred_dist[regions[pred_pos]])
                        return -1 * nll

                    results = minimize(compute_exemplar_nll, [h_old['exemplar_shared']], bounds=((10**-2, 10**2),))
                    h_old['exemplar_shared'] = results.x[0]

                    us_shared_pos = us_shared_inds[chain_pos]
                    uk_shared_pos = uk_shared_inds[chain_pos]
                    aus_shared_pos = aus_shared_inds[chain_pos]

                    if us_shared_pos.shape[0] > 0:
                        us_dists = np.exp(np.concatenate((embed_dists[chain_pos][chain_memstart[chain_pos]:chain_pos][example_regions==0], embed_dists_shared[chain_pos][us_shared_pos])) / h_old['exemplar_opt_shared'])
                    else:
                        us_dists = np.exp(embed_dists[chain_pos][chain_memstart[chain_pos]:chain_pos][example_regions==0] / h_old['exemplar_opt_shared'])

                    if uk_shared_pos.shape[0] > 0:
                        uk_dists = np.exp(np.concatenate((embed_dists[chain_pos][chain_memstart[chain_pos]:chain_pos][example_regions==1], embed_dists_shared[chain_pos][uk_shared_pos])) / h_old['exemplar_opt_shared'])
                    else:
                        uk_dists = np.exp(embed_dists[chain_pos][chain_memstart[chain_pos]:chain_pos][example_regions==1] / h_old['exemplar_opt_shared'])

                    if aus_shared_pos.shape[0] > 0:
                        aus_dists = np.exp(np.concatenate((embed_dists[chain_pos][chain_memstart[chain_pos]:chain_pos][example_regions==2], embed_dists_shared[chain_pos][aus_shared_pos])) / h_old['exemplar_opt_shared'])
                    else:
                        aus_dists = np.exp(embed_dists[chain_pos][chain_memstart[chain_pos]:chain_pos][example_regions==2] / h_old['exemplar_opt_shared'])

                    preds['exemplar_shared'] = int(np.mean(us_dists) < np.mean(uk_dists))

               
            # Collect Results
            for key, value in preds.items():
                if value == target_region:
                    correct_counts_sample[(target_str, n)][key] += 1

model_list = {'Baseline': ['sense_freq', 'sense_freq_shared'],\
                      'Need':['form_need', 'semantic_freq', 'semantic_major', 'context_freq', 'context_major'],\
                      'Simple':['lda', 'lda_shared', 'logistic_reg', 'logistic_reg_shared'],\
                      'Chaining':['1nn', 'prototype', 'exemplar'],\
                      'Chaining - Shared':['1nn_shared', 'prototype_shared', 'exemplar_shared']}

print("%51s%12s%14s" % ("[US]", "[UK]", "[AUS]", "Total"))
for group, models in model_list.items():
    print("["+group.upper()+"]")
    for model in models:

        us_correct = np.asarray([correct_counts_sample[('[US]', n)][model] for n in range(N_trials)])
        us_total = np.asarray([pred_count_sample[('[US]', n)] for n in range(N_trials)])

        uk_correct = np.asarray([correct_counts_sample[('[UK]', n)][model] for n in range(N_trials)])
        uk_total = np.asarray([pred_count_sample[('[UK]', n)] for n in range(N_trials)])

        aus_correct = np.asarray([correct_counts_sample[('[AUS]', n)][model] for n in range(N_trials)])
        aus_total = np.asarray([pred_count_sample[('[AUS]', n)] for n in range(N_trials)])

        print("%35s:   %.1f (%.2f)  %.1f (%.2f)  %.1f (%.2f)" % \
            (model.upper(), \
            np.mean(us_correct / us_total * 100), np.std(us_correct / us_total * 100),\
            np.mean(uk_correct / uk_total * 100), np.std(uk_correct / uk_total * 100),\
            np.mean(aus_correct / aus_total * 100), np.std(aus_correct / aus_total * 100),\

            np.mean((us_correct + uk_correct + aus_correct) / (us_total+uk_total+aus_total) * 100), np.std((us_correct + uk_correct + aus_correct) / (us_total+uk_total+aus_total) * 100)))
    print("")
