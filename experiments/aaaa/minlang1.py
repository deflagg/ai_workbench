import random

nouns = ["cat", "rat", "dog"]
verbs = ["ro", "zi", "xek"]
adjectives = ["nox", "rev", "fii"]
adverbs = ["mol", "tex", "sox"]
determiners = ["ga", "za"]
conjs = ["ul", "ap"]
punct = [".", "?", "!"]

def random_noun_phrase():
    # e.g. 50% chance an article, 30% chance an adjective, etc.
    phrase = []
    if random.random() < 0.5:
        phrase.append(random.choice(determiners))
    if random.random() < 0.3:
        phrase.append(random.choice(adjectives))
    phrase.append(random.choice(nouns))
    return " ".join(phrase)

def random_verb_phrase():
    # Possibly insert an adverb in random positions
    v = random.choice(verbs)
    if random.random() < 0.3:
        # adverb before
        return random.choice(adverbs) + " " + v
    else:
        # adverb after
        if random.random() < 0.3:
            return v + " " + random.choice(adverbs)
        else:
            return v

def random_sentence():
    # simplest pattern: S → Subj V (Obj)
    subj = random_noun_phrase()
    vp = random_verb_phrase()
    sentence = [subj, vp]
    # 50% chance to have an object
    if random.random() < 0.5:
        obj = random_noun_phrase()
        sentence.append(obj)
    return " ".join(sentence)

def maybe_question_or_exclamation(s):
    # 10% chance question, 10% exclamation, else period
    r = random.random()
    if r < 0.1:
        return s + "?"
    elif r < 0.2:
        return s + "!"
    else:
        return s + "."

def random_compound_sentence():
    # 50% chance to compound
    s = random_sentence()
    if random.random() < 0.5:
        s2 = random_sentence()
        conj = random.choice(conjs)
        s = s + " " + conj + " " + s2
    return maybe_question_or_exclamation(s)

# Generate data
for _ in range(20):
    print(random_compound_sentence())
