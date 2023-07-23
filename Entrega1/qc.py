import sys, re, nltk
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from random import randint
stop_words = ["the", "?", "to", "a", "an", ",", "and", "on", "in", "I", "you", "our", "me", "his", "her", "herself", "himself", "she", "he"] #

def remove_stop_words(s):
   words = s.split()

   parsed_string = ""

   for word in words:
      if word.lower() not in stop_words:
         parsed_string += word + " "

   return parsed_string[:-1]

def parse_question(s):
   s = remove_stop_words(s)

   stemmer = WordNetLemmatizer()

   document = re.sub(r'\W', ' ', s)
 
   # Substituting multiple spaces with single space
   document = re.sub(r'\s+', ' ', document, flags=re.I)
    
   # Removing prefixed 'b'
   document = re.sub(r'^b\s+', '', document)
    
   # Converting to Lowercase
   document = document.lower()
    
   # Lemmatization
   document = document.split()

   document = [stemmer.lemmatize(word) for word in document]
   document = ' '.join(document)
    
   return document

def is_acronym(word):
   return len(word) > 1 and (word.isupper() or re.match("('')?[A-Z](\.[A-Z])+\.?('')?", word))

def pattern_identifier(s, model):

   #s = s.replace("'' ", "").replace("`` ", "").replace("\" ", "")

   words = s.split(" ")

   if words[0] == "What" and words[1] == "'s":
      words[1] = "is"

   #Human
   #Description
   if s.startswith("Who is the Queen Mother"):
      return "HUM:desc" if model == "-fine" else "HUM"

   if s.startswith("Who is") or s.startswith("Who 's") or s.startswith("Who was"):
      
      i = 2

      while re.match(r"[A-Z][a-z]*", words[i]):
         i += 1

      if words[i] == "?":
         return "HUM:desc" if model == "-fine" else "HUM"
      elif words[i] == "in":
         i += 1
         while re.match(r"[A-Z][a-z]*", words[i]):
            i += 1

         if words[i] == "?":
            return "HUM:desc" if model == "-fine" else "HUM"

   #Abreviation
   #Expansion
   if re.match(r"What do[es]", s) or s.startswith("What is") or s.startswith("What 's") or s.startswith("What are"):
      if is_acronym(words[2]) and not re.match(r"[A-Z].*", words[3]):
         return "ABBR:exp" if model == "-fine" else "ABBR"

      #if re.match(r".*'' [a-zA-Z]+ ''", s):
      #  return "ABBR:exp" if model == "-fine" else "ABBR"

   if is_acronym(words[0]) and re.match(r".* abbreviation for what", s) and re.match(r".* an acronym for what", s):
      return "ABBR:exp" if model == "-fine" else "ABBR"

   if re.match(r"What .* stand for", s):
      for word in words:
         if word == "stand":
            break
         if is_acronym(word):
            return "ABBR:exp" if model == "-fine" else "ABBR"

   #Abbreviation
   if (re.match(r"What .* the abbreviat", s) and "stand for" in s) or re.match(r"What .* the acronym", s):
      return "ABBR:abb" if model == "-fine" else "ABBR"

   #Entity
   #Term
   if s.endswith("known as ? "):
      return "ENTY:termeq" if model == "-fine" else "ENTY"

   if re.match("What is the former name of", s) or re.match("What is the common name for", s):
      return "ENTY:termeq" if model == "-fine" else "ENTY"

   if re.match(r".* translates to", s) or re.match(r"What is the (.*[^medical] )?term for", s):
      return "ENTY:termeq" if model == "-fine" else "ENTY"

   if (s.startswith("How do I say") or s.startswith("How do you say")) and ("language" in s or re.match(r".* in [A-Z][a-z]*", s)):
      return "ENTY:termeq" if model == "-fine" else "ENTY"

   #Substance
   if (s.startswith("What is") or s.startswith("What 's") or s.startswith("What are") or s.startswith("What was") or s.startswith("What were")) and (s.endswith("made of ? ") or s.endswith("composed of ? ")):
      return "ENTY:substance" if model == "-fine" else "ENTY"

   if s.startswith("What does") and s.endswith("consist of ? "):
      return "ENTY:substance" if model == "-fine" else "ENTY"

   #Animal
   if s.startswith("What species is"):
      return "ENTY:animal" if model == "-fine" else "ENTY"

   #Description
   #Definition
   if re.match(r"What is a?", s) or re.match(r"What is an", s) or s.startswith(r"What 's a?") or s.startswith(r"What 's an") or s.startswith("What are"):

      parsed = words[2:-2]

      if len(parsed) != 0 and (parsed[0] == "a" or parsed[0] == "an"):
         parsed = parsed[1:]

      tagged_sent = nltk.pos_tag(parsed)
      
      if len(parsed) == 1 and tagged_sent[0][1].startswith("NN"):
         return "DESC:def" if model == "-fine" else "DESC"
      elif len(parsed) == 2 and ((tagged_sent[0][1].startswith("JJ") and tagged_sent[1][1].startswith("NN")) or (tagged_sent[0][1].startswith("NN") and tagged_sent[1][1].startswith("NN"))):
         return "DESC:def" if model == "-fine" else "DESC"


   if re.match(r"What .* the difference between", s):
      return "DESC:desc" if model == "-fine" else "DESC"

   #Reason
   if s.startswith("What causes"):
      return "DESC:reason" if model == "-fine" else "DESC"

   #Manner
   if s.startswith("How do") or s.startswith("How does"):
      return "DESC:manner" if model == "-fine" else "DESC"

   #Location
   #Country and City
   for loc in ["countr", "cit"]:

      if s.startswith("What " + loc) or s.startswith("Which " + loc) or s.startswith("In what " + loc) or s.startswith("In which " + loc):
         return "LOC:" + loc + "y" if model == "-fine" else "LOC"

      if s.startswith("What"):
         tagged_sent = nltk.pos_tag(words[1:])

         i = 0

         if tagged_sent[i][0] == "was" or tagged_sent[i][0] == "is" or tagged_sent[i][0] == "are" or tagged_sent[i][0] == "were":
            i += 1

         while tagged_sent[i][1].startswith("JJ") or tagged_sent[i][1].startswith("RB") or tagged_sent[i][1] == "DT" or tagged_sent[i][1] == "NNP" or tagged_sent[i][0] == "'s" or tagged_sent[i][1] == "CD":
            i += 1

         if tagged_sent[i][0].startswith(loc):
            return "LOC:" + loc + "y" if model == "-fine" else "LOC"

         #animal_syn = wn.synsets(tagged_sent[i][0])
         
         #if len(animal_syn) != 0:
         #  animal_syn = animal_syn[0]
         #  hyper = lambda s: s.hypernyms()
         #  if wn.synset('animal.n.01') in list(animal_syn.closure(hyper)):
         #     return "ENTY:animal" if model == "-fine" else "ENTY"

   return None

def train(questions, labels):
   seed = 2624

   txt_clf = Pipeline([ ('vect', CountVectorizer()),('tfidf', TfidfTransformer()), ('clf-svm', SGDClassifier(random_state = seed))])
   parameters_svm = {'vect__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3)],'tfidf__use_idf': [True, False]}
   
   grid_result = GridSearchCV(txt_clf, parameters_svm, n_jobs=-1)
   grid_result.fit(questions, labels)
   best_params = grid_result.best_params_

   return grid_result


def main(argv):

   if len(argv) < 2 or len(argv) > 3:
      print("Expected:\n\t -coarse/-fine train_file\nor\n\t -coarse/-fine train_file DEV_questions_file")
      exit(2)

   model = argv[0]
   train_file = argv[1]
   dev_file = argv[2]

   f_train = open(train_file, "r")
   f_dev = open(dev_file, "r")
   
   questions = []
   labels = []

   # creates a list of [label, question] for each question/label
   for line in f_train:
      question = parse_question(re.compile("[A-Z]*:[a-z]* ").split(line)[1])
      label = line.split(" ")[0] if model == "-fine" else line.split(":")[0]

      questions.append(question)
      labels.append(label)

   d_questions = []
   pattern_labels = []

   for line in f_dev:
      pattern_labels.append(pattern_identifier(line[1:], model))

      d_questions.append(parse_question(line))

   txt_classifier = train(questions, labels)  

   predicted = txt_classifier.predict(d_questions)


   for i in range(len(predicted)):
      if pattern_labels[i] != None:
         print(pattern_labels[i])
      else:
         print(predicted[i])

   f_train.close()
   f_dev.close()


if __name__ == "__main__":
   main(sys.argv[1:])
