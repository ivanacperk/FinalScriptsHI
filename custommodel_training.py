# Import needed libraries.

#%%
#pip install -U 'spacy[transformers,lookups,apple]'
import pandas as pd
from tqdm import tqdm
import spacy
#%%

# Continue to import needed libraries.
# "En" is short for English.
# DocBin is a container that contains data that we want to make more efficient for a computer to understand it.
from spacy.tokens import DocBin
controls = pd.read_csv('data/sp800-53r5-control-catalog.csv') 
nlp = spacy.blank("en")
db = DocBin()
#%%
# Training data is the data we want to feed into the model so it can be interpreted. 
# training_data is the assigned variable name.
# training_data consists of a list that contains strings. Each string has the entity we want to capture.
# The entity we want to capture is assessment.

training_data = [
    ("Conduct penetration testing [Assignment: organization-defined frequency] on [Assignment: organization-defined systems or systems components].", {"entities": [(8,27, "ASSESSMENT")]}),
    ("Automatically audit account creation, modification, enabling, disabling, and removal actions.", {"entities": [(14, 19, "ASSESSMENT")]}),
    ("Identify the types of events that the system is capable of logging in support of the audit function.", {"entities": [(85,90, "ASSESSMENT")]}), 
    ("Assumptions affecting risk assessments, risk responses, and risk monitoring.", {"entities": [(22,38, "ASSESSMENT")]}), 
    ("Constraints affecting risk assessments, risk responses, and risk monitoring.", {"entities": [(23,39, "ASSESSMENT")]}),
    ("Implement an incident handling capability for incidents involving insider threats.", {"entities": [(66,81, "ASSESSMENT")]}),
    ("Coordinate an incident handling capability for insider threats that includes the following organizational entities [Assignment: organization-defined entities].", {"entities": [(47,62, "ASSESSMENT")]}),
]

# With the list above, create a doc object through nlp.
# We also added a label for the entity "assessment"
for text, annot in tqdm(training_data):
    doc = nlp.make_doc(text) 
    ents = [] 
    for start, end, label in annot["entities"]: 
        span = doc.char_span(start, end, label=label, alignment_mode="contract") 
        if span is None: 
            print(f"Skipping entity for {text}") 
        else: 
            ents.append(span) 
            doc.ents = ents 
    db.add(doc) 

#%%

#now save the doc bin object
db.to_disk("./manualtraining.spacy") 

#%%
