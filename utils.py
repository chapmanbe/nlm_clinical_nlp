import pyConTextNLP.itemData as itemData
import numpy as np
import radnlp.view as rview
import radnlp.rules as rules
import radnlp.schema as schema
import radnlp.utils as utils
import radnlp.split as split
import radnlp.classifier as classifier
from IPython.display import display, HTML, Image
from IPython.html.widgets import interact
import ipywidgets as widgets
from radnlp.data import classrslts 
import pandas as pd
import pymysql

colors={"pulmonary_embolism":"blue",
        "critical_finding":"blue",
        "pneumonia":"blue",
        "pneumothorax":"blue",
        "diverticulitis":"blue",
       "definite_negated_existence":"red",
       "probable_negated_existence":"indianred",
       "ambivalent_existence":"orange",
       "probable_existence":"forestgreen",
       "definite_existence":"green",
       "historical":"goldenrod",
       "indication":"Pink",
       "acute":"golden"}

def getOptions():
    """Generates arguments for specifying database and other parameters"""
    options = {}
    options['lexical_kb'] = ["https://raw.githubusercontent.com/chapmanbe/pyConTextNLP/master/KB/lexical_kb_05042016.tsv"]
    options['domain_kb'] = ["https://raw.githubusercontent.com/chapmanbe/pyConTextNLP/master/KB/critical_findings.tsv"]
    options["schema"] = "https://raw.githubusercontent.com/chapmanbe/pyConTextNLP/master/KB/schema2.csv"
    options["rules"] = "https://raw.githubusercontent.com/chapmanbe/pyConTextNLP/master/KB/classificationRules3.csv" 
    return options

def get_kb_rules_schema(options):
    """
    Get the relevant kb, rules, and schema.
    
    """

    _radnlp_rules = rules.read_rules(options["rules"])
    _schema = schema.read_schema(options["schema"])
    
    modifiers = itemData.itemData()
    targets = itemData.itemData()
    for kb in options['lexical_kb']:
        modifiers.extend( itemData.instantiateFromCSVtoitemData(kb) )
    for kb in options['domain_kb']:
        targets.extend( itemData.instantiateFromCSVtoitemData(kb) )
    return {"rules":_radnlp_rules,
            "schema":_schema,
            "modifiers":modifiers,
            "targets":targets}
    

def analyze_report(report, modifiers, targets, rules, schema):
    """
    given an individual radiology report, creates a pyConTextGraph
    object that contains the context markup
    report: a text string containing the radiology reports
    """
    markup = utils.mark_report(split.get_sentences(report),
                         modifiers,
                         targets)
    
    clssfy =   classifier.classify_document_targets(markup,
                                          rules[0],
                                          rules[1],
                                          rules[2],
                                          schema)
    return classrslts(context_document=markup, exam_type="ctpa", report_text=report, classification_result=clssfy)


doc_split=["IMPRESSION:", "INTERPRETATION:", "CONCLUSION:"]

def find_impression(text, split):
    for term in split:
        if term in text:
            return text.split(term)[1]
    return np.NaN

def clean_reports(data):
    """
    given a pandas DataFrame of radiology reports, return a new data set with the impression section
    and filtered by ct and chest as keywords

    data: pandas dataframe with report in the column "text"
    """
    data2 = data.copy()
    data2["impression"] = data.apply(lambda row: find_impression(row["text"], doc_split), axis=1)
    data2 = data2.dropna(axis=0, inplace=False)
    data2["chest"] = data2.apply(lambda x: 'chest' in x["text"].lower() and 'ct' in x["text"].lower(), axis=1)
    data2 = data2[data2["chest"] == True]
    data2 = data2.reset_index()
    return data2[["text", "impression", "code"]]

def view_markup(reports, colors):
    @interact(i=widgets.IntSlider(min=0, max=len(reports)-1))
    def _view_markup(i):
        markup = reports["pe rslt"][i]
        rview.markup_to_pydot(markup)
        display(Image("tmp.png"))
        mt = rview.markup_to_html(markup, color_map=colors)

        display(HTML(mt))

def get_data():
    conn = pymysql.connect(host="mysql",
                        port=3306,user="jovyan",
                        passwd='jovyan',db='mimic2')

    data = \
        pd.read_sql("""SELECT noteevents.subject_id, 
                      noteevents.text, 
                      icd9.code 
               FROM noteevents INNER JOIN icd9 ON 
                      noteevents.subject_id = icd9.subject_id 
               WHERE (   icd9.code LIKE '415.1%'
                      ) 
                      AND noteevents.category = 'RADIOLOGY_REPORT'""",
            conn).drop_duplicates()
    data = clean_reports(data)
    return data

