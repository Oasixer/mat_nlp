import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.corpus import wordnet as wn
import random

def tag(sentence):
  words = word_tokenize(sentence)
  words = pos_tag(words)
  return words

def paraphraseable(tag):
  return tag.startswith('NN') or tag == 'VB' or tag.startswith('JJ')

def pos(tag):
  if tag.startswith('NN'):
    return wn.NOUN
  elif tag.startswith('V'):
    return wn.VERB

def get_synonym(word, tag):
  synsets = wn.synsets(word, pos(tag))
  lemmas = []
  print(word)
  print(synsets)
  for sense in synsets:
    for lemma in sense.lemmas():
      lemmas.append(lemma.name())

  print(lemmas)
  print("\n")
  
  return random.choice(lemmas)

def paraphrase(sentence):
  new_sentence = []
  for (word, t) in tag(sentence):
    if paraphraseable(t):
      syn = get_synonym(word, t)
      if syn:
        new_sentence.append(syn)
      else:
        new_sentence.append(word)
    else:
      new_sentence.append(word)
  return new_sentence

if __name__ == '__main__':
  #nltk.download('wordnet')
  #nltk.download('averaged_perceptron_tagger')
  test = paraphrase("The quick brown fox happened upon a source for useful information.")
  print (test)