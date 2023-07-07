# Import needed libaries.
# Begin NLP intialization.

#%%
import spacy
from spacy.pipeline import EntityRuler
from spacy import displacy
import pandas as pd
from pprint import pprint

#%%
# Continue to import needed libaries.
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
# Begin OSCAL initiliazation.
# Import needed libaries.
# Import controls we want to capture entities from.
import lib.oscal_helper as oh
oscal = oh.OSCALHelper()
url = "https://raw.githubusercontent.com/usnistgov/oscal-content/main/nist.gov/SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json"
number_of_controls = 75



#%%
# Create variable named controls to get a random sample.
controls = oscal.get_controls_from_json(url, number_of_controls)

#%%
# Create variable named oscal concent to extract the controls.
oscal_content = oscal.get_prose_from_controls(controls)
# Sample () function does not work on Doc.
# Must create a short cut (pandas dataframe).
# df1 is the variable for the dataframe.
# I want to print out a random sample of 75 controls that only contain the Control ID and Control Text.
df1 = pd.DataFrame(oscal_content, columns=['Control_ID', 'Control_Text'])
oscal_content = df1.sample(75)
print(oscal_content)

# End OSCAL intilization. 

#%% 
# Loop through the controls.
if True:
    for index, control in oscal_content.iterrows():
        print(control['Control_ID'])

        doc = nlp(control['Control_Text'].lower())
        # for ent in doc.ents:
        #     print(ent.text, ent.label_)
        displacy.render(doc, style="ent")

#if False - it will not execute, will skip this block/cell/nlp part
#%%


