# Project: Building a Prototype Poetry Generator based on Markov Model 
# Goal: to generate poems that assemble the works of Edgar Allan Poe or Robert Frost poems 
# Dataset: https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/edgar_allan_poe.txt
# https://raw.githubusercontent.com/lazyprogrammer/machine_learning_examples/master/hmm_class/robert_frost.txt

import numpy as np
import string

np.random.seed(1234)

initial = {} # start of a phrase
first_order = {} # second word only
second_order = {}

def remove_punctuation(s):
    return s.translate(str.maketrans('','',string.punctuation))

def add2dict(d, k, v):
  if k not in d:
    d[k] = []
  d[k].append(v)

# [cat, cat, dog, dog, dog, dog, dog, mouse, ...]

for line in open('datasets/robert_frost.txt'):
# for line in open('datasets/edgar_allan_poe.txt'):
  tokens = remove_punctuation(line.rstrip().lower()).split()

  T = len(tokens)
  for i in range(T):
    t = tokens[i]
    if i == 0:
      # measure the distribution of the first word
      initial[t] = initial.get(t, 0.) + 1
    else:
      t_1 = tokens[i-1]
      if i == T - 1:
        # measure probability of ending the line
        add2dict(second_order, (t_1, t), 'END')
      if i == 1:
        # measure distribution of second word
        # given only first word
        add2dict(first_order, t_1, t)
      else:
        t_2 = tokens[i-2]
        add2dict(second_order, (t_2, t_1), t)

# normalize the distributions
initial_total = sum(initial.values())
for t, c in initial.items():
    initial[t] = c / initial_total

# convert [cat, cat, cat, dog, dog, dog, dog, mouse, ...]
# into {cat: 0.5, dog: 0.4, mouse: 0.1}

def list2pdict(ts):
  # turn each list of possibilities into a dictionary of probabilities
  d = {}
  n = len(ts)
  for t in ts:
    d[t] = d.get(t, 0.) + 1
  for t, c in d.items():
    d[t] = c / n
  return d

for t_1, ts in first_order.items():
  # replace list with dictionary of probabilities
  first_order[t_1] = list2pdict(ts)

for k, ts in second_order.items():
  second_order[k] = list2pdict(ts)

def sample_word(d):
  # print "d:", d
  p0 = np.random.random()
  # print "p0:", p0
  cumulative = 0
  for t, p in d.items():
    cumulative += p
    if p0 < cumulative:
      return t
  assert(False) # should never get here

def generate():
  for i in range(4): # generate 4 lines
    sentence = []

    # initial word
    w0 = sample_word(initial)
    sentence.append(w0)

    # sample second word
    w1 = sample_word(first_order[w0])
    sentence.append(w1)

    # second-order transitions until END
    while True:
      w2 = sample_word(second_order[(w0, w1)])
      if w2 == 'END':
        break
      sentence.append(w2)
      w0 = w1
      w1 = w2
    print(' '.join(sentence))

generate()