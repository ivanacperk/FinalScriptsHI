# Import needed libaries. 

#%%
#pip install -U 'spacy[transformers,lookups,apple]'
import pandas as pd
controls = pd.read_csv('data/sp800-53r5-control-catalog.csv') 
from tqdm import tqdm
import spacy
#%%
#Continue to import needed libaries.
# En is short for English.
# DocBin is a container that contains data that we want to make more efficient for a computer to understand it.
from spacy.tokens import DocBin 
nlp = spacy.blank("en")
db = DocBin()

#%%
# Test Data is the data that will be used to test how well the model is interpreting both the training and test data.
# test_data is the assigned variable name.
# test_data consists of a list that contains strings. Each string has a entity that we want to capture.
# The entity we want to capture is assessment.

test_data = [
    ("Require the developer of the system, system component, or system service to perform penetration testing.", {"entities": [(84,103, "ASSESSMENT")]}),
    ("Ensure that audit records contain information that establishes what type of event occured.", {"entities": [(12, 25, "ASSESSMENT")]}),
    ("Audit changes to attributes.", {"entities": [(0,5, "ASSESSMENT")]}), 
    ("Conduct a risk assessment, including identifying threats to and vulnerabilities in the system.", {"entities": [(10,25, "ASSESSMENT")]}), 
    ("Disseminate risk assessment results to [Assignment: organization-defined personnel or roles].", {"entities": [(12,27, "ASSESSMENT")]}),
    ("Explicit focus on handling incidents involving insider threats provides additional emphasis on this type of threat and the need for specific incident handling capabilities to provide appropiate and timely responses.", {"entities": [(47,62, "ASSESSMENT")]}),
]

# With the list above, create a doc through nlp.
# We also added a label for the entity "assessment". 

for text, annot in tqdm(test_data): 
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
#Now save the doc bin object.
db.to_disk("./manualtest.spacy") 

#%%