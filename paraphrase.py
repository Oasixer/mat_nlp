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

def synonyms(word, tag):
  synsets = wn.synsets(word, pos(tag))
  lemmas = []
  for sense in synsets:
    for lemma in sense.lemmas():
      lemmas.append(lemma.name())

  print(lemmas)
  
  return set(lemmas)

def synonymIfExists(sentence):
  for (word, t) in tag(sentence):
    if paraphraseable(t):
      syns = synonyms(word, t)
      if syns:
        if len(syns) > 1:
          yield [word, list(syns)]
          continue
    yield [word, []]

def paraphrase(sentence):
  return [x for x in synonymIfExists(sentence)]

if __name__ == '__main__':
  #nltk.download('wordnet')
  #nltk.download('averaged_perceptron_tagger')
  stringy = '''A variety of properties describe the way in which materials respond to the application of heat. The heat capacity indicates the amount of heat necessary to raise the temperature of a given amount of material. The term specific heat is used when the property is determined for a unit mass of the material. Fundamental understanding of the mechanism of heat absorption by atomic vibrations leads to a useful rule of thumb for estimating heat capacity of materials at room temperature and above (Cp≈CV≈3R)(Cp≈CV≈3R).

The increasing vibration of atoms with increasing temperature leads to increasing interatomic separations and, generally, a positive coefficient of thermal expansion. A careful inspection of the relationship of this expansion to the atomic bonding energy curve reveals that strong bonding correlates with low thermal expansion as well as high elastic modulus and high melting point.

Heat conduction in materials can be described with the thermal conductivity, k , in the same way as mass transport was described in Chapter 5 using the diffusivity, D. The mechanism of thermal conductivity in metals is largely associated with their conductive electrons, whereas the mechanism for ceramics and polymers is largely associated with atomic vibrations. Due to the wavelike nature of both mechanisms, increasing temperature and structural disorder both tend to diminish thermal conductivity. Porosity is especially effective in diminishing thermal conductivity.

The inherent brittleness of ceramics and glasses, combined with thermal-expansion mismatch or low thermal conductivities, can lead to mechanical failure by thermal shock. Sudden cooling is especially effective in creating excessive surface tensile stress and subsequent fracture.'''
  test = [random.choice(word_list) for word_list in paraphrase(stringy)]
  print (test)