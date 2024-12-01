import bs4
import pickle
import re
import glob
import numpy as np
from tqdm import trange

from util import GSD_Definition, GSD_Word
from process import process_GSD

word_hash = [s[:-5] for s in glob.glob('*.html')]

process_GSD(word_hash, input_dir = "", output_dir = "")

data = [pickle.load(open(h+'.pickle', 'rb')) for h in word_hash]

#Original Version
regions = ['[US]', '[UK]']

#New Version accounting for Australian Slang
#regions = ['[US]', '[UK]', '[Aus]']

punctuations = '!\'"#$%&()\*\+,-\./:;<=>?@[\\]^_`{|}~'

re_punc = re.compile(r"["+punctuations+r"]+")
re_space = re.compile(r" +")

re_extract_quote = re.compile(r"[1-9/]+:")
re_extract_quote_all = re.compile(r"[1-9/]+:.*$")

def proc_quote_sent(sent):
    return re_extract_quote.sub(' ', re_extract_quote_all.findall(sent)[0]).strip()

def validate_quote_sent(word, sent):
    tokens = [s.lower() for s in re_space.sub(' ', re_punc.sub('', sent)).split(' ')]
    return word.lower() in tokens

data_proc = []

for i in trange(len(data)):
    w = data[i]
    if w.is_abbr():
        continue
    d_list = []
    for d in w.definitions:
        stamps = d.stamps
        region_set = set([s[1] for s in stamps])
        if np.any([r in region_set for r in regions]):
            new_stamps = [s for s in stamps if np.any([r==s[1] in region_set for r in regions])]
            new_def = GSD_Definition(d.def_sent)
            new_def.stamps = new_stamps
            new_def.contexts = {key:value for key, value in d.contexts.items() if key in new_stamps}
            d_list.append(new_def)
    if len(d_list) > 0:
        new_word = GSD_Word(w.word.replace("\\xe2\\x80\\x99", "'").replace("\\xe2\\x80\\x98", "'"), w.pos, w.homonym)
        new_word.definitions = d_list
        data_proc.append(new_word)

_ = [print(d) for d in data_proc]

np.save('GSD_sample_data.npy', data_proc)
