# Import needed libaries.
# Begin NLP intialization.

#%%
import spacy
from spacy.pipeline import EntityRuler
from spacy import displacy
import pandas as pd
from pprint import pprint

#%%
# Continue to import needed libaries
import lib.helpers as h
helper = h.Helper()
# Patterns is the variable name of the list.
# Patterns consist of a list of entities we want to capture from the controls.
patterns = []
patterns.extend(helper.get_patterns_for_entity("access control"))
patterns.extend(helper.get_patterns_for_entity("account"))
patterns.extend(helper.get_patterns_for_entity("assessment"))
patterns.extend(helper.get_patterns_for_entity("software"))
patterns.extend(helper.get_patterns_for_entity("address"))
patterns.extend(helper.get_patterns_for_entity("security control"))
#%%
# Print out the list of patterns.
pprint(patterns)

#%%
# Load new spaCy model. 
nlp = spacy.load("en_core_web_trf")

#%% 
# Add the entity ruler to the pipeline before the default NER component.
nlp.add_pipe("entity_ruler", before="ner")

#%% 
# Create an EntityRuler. ER allows you to add your own entities to identify.
entity_ruler = nlp.get_pipe("entity_ruler")

#%% 
# Add patterns to the entity ruler.
entity_ruler.add_patterns(patterns)

#%%
# Create test data
GTest_Data = [
    ("A reference monitor is a set of design requirements on a reference validation mechanism that, as a key component of an operating system, enforces an access control policy over all subjects and objects.", {"entities": [(119,135, "SOFTWARE")]}),
    ("System-level information includes system state information, operating system software, middleware, application software, and licenses.", {"entities": [(61, 86, "SOFTWARE")]}),
    ("Program management controls may be implemented at the organization level or the mission or business process level, and are essential for managing the organizations information security program.", {"entities": [(0, 27, "SECURITY CONTROL")]}),
    ("To reduce the cost of reauthorization, authorizing officials can leverage the results of continuous monitoring processes to the maximum extent possible as the basis for rendering reauthorization decisions.", {"entities": [(89, 110, "SECURITY CONTROL")]}),
    ("Continuous monitoring at the system level facilitates ongoing awareness of the system security and privacy posture to support organizational risk management decisions.", {"entities": [(0, 21, "SECURITY CONTROL")]}),
    ("Continuous monitoring at the system level facilitates ongoing awareness of the system security and privacy posture to support organizational risk management decisions.", {"entities": [(99, 115, "SOFTWARE")]}),
    ("Access control policy and procedures address the controls in the AC family that are implemented within systems and organizations.", {"entities": [(0, 21, "ADDRESS")]}),
    ("The risk management strategy is an important factor in establishing such policies and procedures.", {"entities": [(4, 28, "ADDRESS")]}),
    ("Penetration testing can be conducted internally or externally on the hardware, software, or firmware components of a system and can exercise both physical and technical controls.", {"entities": [(0, 19, "ASSESSMENT")]}),
    ("Insider threat programs can leverage the existence of incident handling teams that organizations may already have in place, such as computer security incident response teams.", {"entities": [(0, 23, "ASSESSMENT")]}),
    ("Before permitting the use of shared or group accounts, organizations consider the increased risk due to the lack of accountability with such accounts.", {"entites": [(39, 53, "ACCOUNT")]}),
    ("Individual authentication prior to shared group authentication mitigates the risk of using group accounts or authenticators.", {"entities": [(91, 105, "ACCOUNT")]}),
    ("Security functions include establishing system accounts, configuring access authorizations (i.e., permissions, privileges), configuring settings for events to be audited, and establishing intrusion detection parameters.", {"entities": [(40, 55, "ACCOUNT")]}),
    ("Security marking refers to the application or use of human-readable security attributes.", {"entities": [(68, 87, "ACCESS CONTROL")]}),
    ("The enforcement of trusted communications paths is provided by a specific implementation that meets the reference monitor concept.", {"entities": [(104, 121, "ACCESS CONTROL")]}),
]

#%%
#%% 
# Loop through the test data
# Now compare how well it identified entities versus the glossary approach.
for text, annot in GTest_Data:
        doc = nlp(text)
        displacy.render(doc, style="ent")
#%%

