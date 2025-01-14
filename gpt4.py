import datetime
import os
import zipfile
from openai import OpenAI
import sqlite3
import urllib.request
import wn
from datetime import datetime
from collections import Counter
import pandas as pd

PROMPT_TEMPLATE_LEMMA = "The word '{LEMMA}' is a {POS} in English with the definition '{DEFINITION}'. Translate the word '{LEMMA}' into {LANGUAGE}, making sure that the correct meaning in context is conveyed. Give only the translated word in your answer."
PROMPT_TEMPLATE_DEFINITION = "The word '{LEMMA}' is a {POS} in English with the definition '{DEFINITION}'. Translate the definition into {LANGUAGE}, making sure that the correct meaning in context is conveyed. Give only the translated definition in your answer."

PROMPT_TEMPLATE_LEMMA_AF = "The word '{LEMMA}' is a {POS} in English with the definition '{DEFINITION}'. Translate the word '{LEMMA}' into Afrikaans, making sure that the correct meaning in context is conveyed. Translate the word, and give only a comma-seperated list of the translated word and its synonyms in your answer."
PROMPT_TEMPLATE_LEMMA_AF_DOUBLE_SHOT = "Which one of the following Afrikaans phrases refering to the English {POS} '{LEMMA}' (with the meaning '{DEFINITION}') is grammatically the most correct: {OPTIONS}. Give only the most correct phrase in your answer."

PROMPT_TEMPLATE_LEMMA_JUDGE = "The word '{LEMMA}' is a {POS} in English with the definition '{DEFINITION}'. Translate the word '{LEMMA}' into {LANGUAGE}, making sure that the correct meaning in context is conveyed. Translate the word, and give only a comma-seperated list of the translated word and its synonyms in your answer."
PROMPT_TEMPLATE_LEMMA_JUDGE_2 = "Which one of the following {LANGUAGE} phrases refering to the English {POS} '{LEMMA}' (with the meaning '{DEFINITION}') is grammatically the most correct: {OPTIONS}. Give only the most correct phrase in your answer."

text_client = OpenAI(
    #api_key=os.environ.get("OPENAI_API_KEY"),
    api_key=""
)

pos_description = {}
pos_description['n'] = "noun"
pos_description['v'] = "verb"
pos_description['a'] = "adjective"
pos_description['s'] = "adjective"
pos_description['r'] = "adverb"

LLM_RESULT_TABLE = """
    CREATE TABLE LLM_RESULT
        (          
        ILI          TEXT    NOT NULL,          
        POS          TEXT    NOT NULL,          
        LEMMA_EN          TEXT    NOT NULL,
        DECRIPTION_EN          TEXT    NOT NULL,
        LEMMA_TARGET          TEXT    NOT NULL,
        DEFINITION_TARGET          TEXT    NOT NULL,         
        LEMMA_LLM           TEXT     NOT NULL,
        DEFINITION_LLM      TEXT     NOT NULL,
        LANG      TEXT     NOT NULL,
        MODEL      TEXT     NOT NULL,
        TIMESTAMP      TEXT     NOT NULL
        );
    """
LLM_RESULT_TABLE_INSERT = """
INSERT INTO LLM_RESULT 
(ILI,      
POS,               
LEMMA_EN,
DECRIPTION_EN,
LEMMA_TARGET,
DEFINITION_TARGET,         
LEMMA_LLM,
DEFINITION_LLM,
LANG,
MODEL,
TIMESTAMP) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

EVAL_TABLE = """
    CREATE TABLE EVAL
        (          
        ILI          TEXT    NOT NULL,          
        POS          TEXT    NOT NULL,          
        LEMMA_EN          TEXT    NOT NULL,
        DECRIPTION_EN          TEXT    NOT NULL,
        LEMMA_TARGET          TEXT    NOT NULL,
        DEFINITION_TARGET          TEXT    NOT NULL,         
        LEMMA_LLM           TEXT     NOT NULL,
        DEFINITION_LLM      TEXT     NOT NULL,
        TRANS_LEMMA_ONLY      TEXT     NOT NULL,
        LANG      TEXT     NOT NULL,
        ORIG_TARGET_LEMMAS      TEXT     NOT NULL,
        COMMENTS      TEXT     NOT NULL
        );
    """
EVAL_TABLE_INSERT = """
INSERT INTO EVAL 
(ILI,      
POS,               
LEMMA_EN,
DECRIPTION_EN,
LEMMA_TARGET,
DEFINITION_TARGET,         
LEMMA_LLM,
DEFINITION_LLM,
TRANS_LEMMA_ONLY,
LANG,
ORIG_TARGET_LEMMAS,
COMMENTS) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

EVAL_AF_2_TABLE_INSERT = """
INSERT INTO AF_EVAL2 
(ILI,      
POS,               
LEMMA_EN,
DECRIPTION_EN,
LEMMA_LLM,
COMMENTS,
LEMMA_LLM2,
LEMMA_LLM2_OPTIONS,
COMMENTS2) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

EVAL_DE_MISMATCH_MERGE = """
Select distinct * from (
SELECT ILI, POS, LEMMA_EN, DECRIPTION_EN, LEMMA_TARGET AS EVAL_LEMMA
FROM EVAL
WHERE ORIG_TARGET_LEMMAS NOT LIKE '%' || LEMMA_TARGET || '%'
UNION
SELECT ILI, POS, LEMMA_EN, DECRIPTION_EN, LEMMA_LLM AS EVAL_LEMMA
FROM EVAL
WHERE ORIG_TARGET_LEMMAS NOT LIKE '%' || LEMMA_LLM || '%'
UNION
SELECT ILI, POS, LEMMA_EN, DECRIPTION_EN, TRANS_LEMMA_ONLY AS EVAL_LEMMA
FROM EVAL
WHERE ORIG_TARGET_LEMMAS NOT LIKE '%' || TRANS_LEMMA_ONLY || '%')
"""

EVAL_DE_MISMATCH_TABLE_INSERT = """
INSERT INTO EVAL_MISMATCH
(ILI,      
POS,               
LEMMA_EN,
DECRIPTION_EN,
DESCRIPTION_ODENET,
EVAL_LEMMA,
RESULT,
COMMENTS) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
"""

EVAL_DE_MISMATCH_TABLE_UPDATE = """
UPDATE EVAL_MISMATCH 
SET RESULT = ?
WHERE ILI = ?
AND LEMMA_EN = ?
AND EVAL_LEMMA = ?
"""

LLM_AS_A_JUDGE_INSERT = """
INSERT INTO LLM_AS_A_JUDGE
(ILI,
POS,
LEMMA_EN,
DECRIPTION_EN,	
DEFINITION_TARGET,
LEMMA_PROMPT_1,
LEMMA_PROMPT_2,
LEMMA_PROMPT_RESPONSE,
LEMMA_PROMPT2_OPTIONS,
RESULT,
COMMENTS,
LEMMAS_TARGET) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
"""

LLM_AS_A_JUDGE_UPDATE = """
UPDATE LLM_AS_A_JUDGE 
SET RESULT = ?
WHERE ILI = ?
AND LEMMA_EN = ?
AND LEMMA_PROMPT_RESPONSE = ?
"""

def create_result_table(db_name):
    try:
        conn = sqlite3.connect(db_name)
        cursor = conn.cursor()
        cursor.execute(EVAL_TABLE)
        conn.commit()
        cursor.execute(LLM_RESULT_TABLE)
        conn.commit()
        conn.close()
    except Exception as e:
        print(e)    

def save_results(db_name, query, results):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.executemany(query, results)
    conn.commit()
    conn.close()
    
    
def get_ilis_with_conf_1_0() -> list:
    ILIS_CONF_SCORE_1 = """
        select ilis.id from synsets 
        INNER JOIN ilis on ilis.rowid = synsets.ili_rowid
        where CAST(synsets.metadata as TEXT) like '%"confidenceScore": "1.0"%'
    """
    conn = sqlite3.connect(os.path.join(os.path.expanduser("~"), '.wn_data', 'wn.db'))
    cursor = conn.cursor()
    cursor.execute(ILIS_CONF_SCORE_1)    
    rows = cursor.fetchall()    
    ilis = [row[0] for row in rows]        
    conn.close()
    return ilis

def get_llm_synset_target(ilis:list):
    results = []    
    langs = [('de', 'German'), ('af', 'Afrikaans')]
    timestamp = str(datetime.now())
    for index, (lang, lang_description) in enumerate(langs):
        for index, ili in enumerate(ilis):    
            synsets = wn.synsets(ili=ili, lang='en')
            if len(synsets) > 0:
                try:
                    lemma_en = synsets[0].lemmas()[0]
                    definition_en = synsets[0].definition()        
                    definition_prompt = PROMPT_TEMPLATE_DEFINITION.replace('{LEMMA}', lemma_en).replace('{POS}', pos_description[synsets[0].pos]).replace('{DEFINITION}', definition_en).replace('{LANGUAGE}', lang_description)
                    lemma_prompt = PROMPT_TEMPLATE_LEMMA.replace('{LEMMA}', lemma_en).replace('{POS}', pos_description[synsets[0].pos]).replace('{DEFINITION}', definition_en).replace('{LANGUAGE}', lang_description)
                    
                    response = text_client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "system", "content": 'You are a linguistic assistant, providing answers regarding language-related questions.'},
                                    {"role": "user", "content": lemma_prompt}
                        ])
                    lemma_llm = response.choices[0].message.content
                    lemma_llm = lemma_llm.replace('\n', ' ').strip()                                                        
                    
                    response = text_client.chat.completions.create(
                        model="gpt-4",
                        messages=[{"role": "system", "content": 'You are a linguistic assistant, providing answers regarding language-related questions.'},
                                    {"role": "user", "content": definition_prompt}
                        ])
                    definition_llm = response.choices[0].message.content
                    definition_llm = definition_llm.replace('\n', ' ').strip()                                                    
                    
                    synsets = wn.synsets(ili=ili, lang=lang)        
                    lemma_target = synsets[0].lemmas()[0]
                    definition_target = synsets[0].definition()        
                    results.append((synsets[0].ili.id, synsets[0].pos, lemma_en, definition_en, lemma_target, definition_target, lemma_llm, definition_llm, lang, "GPT-4", timestamp))
                except Exception as e:
                    print(f'Error at index {index} for ili {synsets[0].ili.id}: {e}')

                if (index % 100 == 0):
                    print(f'{index} of {len(ilis)} records processed ...')
                    save_results('results.db', LLM_RESULT_TABLE_INSERT, results)
                    results = []

        save_results('results.db', LLM_RESULT_TABLE_INSERT, results)
        results = []        

def preprocessing():
    try:
        wn.download('ewn:2020')
        wn.download('odenet:1.4')
        urllib.request.urlretrieve('https://github.com/pssvlng/open-afrikaans-wordnet/raw/main/OAfrikaansNet.xml', 'OAfrikaansNet.xml')
        wn.add('OAfrikaansNet.xml')
    except Exception as e:
        print(e)    

def preprocessing_with_de_inferred():
    try:
        wn.download('ewn:2020')
        urllib.request.urlretrieve('https://github.com/pssvlng/open-european-WordNets-inferred/raw/main/de_inferred_plus.zip', 'de_inferred_plus.zip')
        with zipfile.ZipFile('de_inferred_plus.zip', 'r') as zip_ref:
            zip_ref.extractall('')
        wn.add('de_inferred_plus.xml')
        urllib.request.urlretrieve('https://github.com/pssvlng/open-afrikaans-wordnet/raw/main/OAfrikaansNet.xml', 'OAfrikaansNet.xml')
        wn.add('OAfrikaansNet.xml')
    except Exception as e:
        print(e)  

def copy_results(source_db, target_db, lang):
    conn = sqlite3.connect(source_db)    
    cursor = conn.cursor()
    cursor.execute(f"Select * from LLM_RESULT where lang='{lang}'")    
    rows = cursor.fetchall()    
    results = []
    for row in rows:
        ili = str(row[0])
        synsets = wn.synsets(ili=ili, lang=lang)
        synsets[0]
        results.append((synsets[0].ili.id, synsets[0].pos, row[2], row[3], synsets[0].lemmas()[0], synsets[0].definition(), row[6], row[7], lang, "GPT-4", row[10]))
    save_results(target_db, LLM_RESULT_TABLE_INSERT, results)    
    conn.close()

def get_lemma_trans(filename: str) ->dict:     
    result = {}
    with open(filename, 'r') as file:
        for line in file:
            values = line.split(': ')
            result[values[0]] = values[1]

    return result        

def populate_eval_table(db_name:str, trans_lemmas:dict, lang):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("Select * from LLM_RESULT")    
    rows = cursor.fetchall()    
    results = []
    for row in rows:
        ili = str(row[0])
        synsets = wn.synsets(ili=ili, lang=lang)
        lemmas = ', '.join(synsets[0].lemmas())
        trans_only = trans_lemmas[ili]
        results.append((ili, row[1], row[2], row[3], row[4], row[5], row[6], row[7], trans_only, lang, lemmas, ""))

    conn.close()
    save_results(db_name, EVAL_TABLE_INSERT, results)    


def llm_as_a_judge(ilis:list):
    results = []    
    #langs = [('de', 'German', 'resutls_de.db'), ('af', 'Afrikaans', 'results_af.db')]
    langs = [('af', 'Afrikaans', 'results_af.db')]    
    for index, (lang, lang_description, db_name) in enumerate(langs):
        for index, ili in enumerate(ilis):    
            synsets = wn.synsets(ili=ili, lang='en')
            if len(synsets) > 0:
                try:
                    lemma_en = synsets[0].lemmas()[0]      
                    definition_en = synsets[0].definition()              
                    pos = synsets[0].pos
                    lemma_prompt = PROMPT_TEMPLATE_LEMMA_JUDGE.replace('{LEMMA}', lemma_en).replace('{POS}', pos_description[synsets[0].pos]).replace('{DEFINITION}', definition_en).replace('{LANGUAGE}', lang_description)
                    
                    try_list = []
                    for i in range(3):
                        response = text_client.chat.completions.create(
                            model="gpt-4",
                            messages=[{"role": "system", "content": 'You are a linguistic assistant, providing answers regarding language-related questions.'},
                                        {"role": "user", "content": lemma_prompt}
                            ])
                        lemma_llm = response.choices[0].message.content                        
                        
                        lemma_llm = lemma_llm.replace('\n', ' ').strip()                
                        lemma_list = [item.replace('"', '').replace("'", "").strip().lower() for item in lemma_llm.split(',')]
                        try_list.append(lemma_list)                                                            
                                                                                
                    merged_list = [item for sublist in try_list for item in sublist]
                    options = ''
                    for idx, item in enumerate(list(set(merged_list))):
                        options = options + f'{idx + 1}) {item}  '                 

                    double_shot_prompt = PROMPT_TEMPLATE_LEMMA_JUDGE_2.replace('{LEMMA}', lemma_en).replace('{DEFINITION}', definition_en).replace('{POS}', pos_description[pos]).replace('{OPTIONS}', options.strip()).replace('{LANGUAGE}', lang_description)
                    response = text_client.chat.completions.create(
                                    model="gpt-4",
                                    messages=[{"role": "system", "content": 'You are a linguistic assistant, providing answers regarding language-related questions.'},
                                                {"role": "user", "content": double_shot_prompt}
                                    ])
                    double_shot_response = response.choices[0].message.content.strip()                    

                    formatted_try_list = []
                    for entry in try_list:
                        formatted_entry = f"[{', '.join(entry)}]"
                        formatted_try_list.append(formatted_entry)

                    stats = str(Counter(merged_list))
                    formatted_try_list.append(stats)

                    synsets = wn.synsets(ili=ili, lang=lang)
                    definition_target = synsets[0].definition()             
                    lemmas_target = synsets[0].lemmas()
                    result = ''
                    comment = ''
                    #if double_shot_response in lemmas_target:
                    #    result = 'J'
                    #    comment = 'ODENET'
                    results.append((synsets[0].ili.id, synsets[0].pos, lemma_en, definition_en, definition_target, lemma_prompt, double_shot_prompt, double_shot_response, ', '.join(formatted_try_list), result, comment, ', '.join(lemmas_target)))
                except Exception as e:
                    print(f'Error at index {index} for ili {synsets[0].ili.id}: {e}')

                if (index % 100 == 0):
                    print(f'{index} of {len(ilis)} records processed ...')
                    save_results(db_name, LLM_AS_A_JUDGE_INSERT, results)
                    results = []

        save_results(db_name, LLM_AS_A_JUDGE_INSERT, results)
        results = []      

def re_eval_af(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("select * from AF_EVAL where COMMENTS like '%-%' or COMMENTS like '%<>%'")    
    rows = cursor.fetchall()    
    results = []
    for index, row in enumerate(rows):
        try_list = []                
        ili = str(row[0])
        pos = str(row[1])
        lemma_en = str(row[2])
        description_en = str(row[3])
        lemma_llm = str(row[5])
        comments = str(row[6])
        prompt_af = PROMPT_TEMPLATE_LEMMA_AF.replace('{LEMMA}', lemma_en).replace('{DEFINITION}', description_en).replace('{POS}', pos_description[pos])
        
        try:        
            for i in range(3):
                response = text_client.chat.completions.create(
                            model="gpt-4",
                            messages=[{"role": "system", "content": 'You are a linguistic assistant, providing answers regarding language-related questions.'},
                                        {"role": "user", "content": prompt_af}
                            ])
                lemma_llm2 = response.choices[0].message.content
                lemma_llm2 = lemma_llm2.replace('\n', ' ').strip()                
                lemma_list = [item.replace('"', '').replace("'", "").strip().lower() for item in lemma_llm2.split(',')]
                try_list.append(lemma_list)                                                            

            merged_list = [item for sublist in try_list for item in sublist]
            options = ''
            for idx, item in enumerate(list(set(merged_list))):
                options = options + f'{idx + 1}) {item}  '                 

            double_shot_prompt = PROMPT_TEMPLATE_LEMMA_AF_DOUBLE_SHOT.replace('{LEMMA}', lemma_en).replace('{DEFINITION}', description_en).replace('{POS}', pos_description[pos]).replace('{OPTIONS}', options.strip())
            response = text_client.chat.completions.create(
                            model="gpt-4",
                            messages=[{"role": "system", "content": 'You are a linguistic assistant, providing answers regarding language-related questions.'},
                                        {"role": "user", "content": double_shot_prompt}
                            ])
            double_shot_response = response.choices[0].message.content

            formatted_try_list = []
            for entry in try_list:
                formatted_entry = f"[{', '.join(entry)}]"
                formatted_try_list.append(formatted_entry)

            stats = str(Counter(merged_list))
            formatted_try_list.append(stats)
            results.append((ili, pos, lemma_en, description_en, lemma_llm, comments, double_shot_response, ', '.join(formatted_try_list), ""))

        except Exception as e:
            print(e)

    conn.close()
    save_results(db_name, EVAL_AF_2_TABLE_INSERT, results)    


def mismatch_eval_de(db_name):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute(EVAL_DE_MISMATCH_MERGE)    
    rows = cursor.fetchall()    
    results = []
    for row in rows:
        ili = str(row[0])
        synsets = wn.synsets(ili=ili, lang='de')
        definition = synsets[0].definition()
        if str(row[1]) != synsets[0].pos and str(row[1]) not in ['s', 'a', 'r']:
            print(f'POS mismatch: ili: {ili} - {definition} {row[1]} vs. {synsets[0].pos}')
        
        
        results.append((ili, pos_description[row[1]], row[2], row[3], definition, row[4], "", ""))

    conn.close()
    save_results(db_name, EVAL_DE_MISMATCH_TABLE_INSERT, results)    

def import_eval(db_name, file_name, query):    
    df = pd.read_csv(file_name)    
    records_as_tuples = list(df.itertuples(index=False, name=None))
    rearranged_tuples = [(t[-1], *t[:-1]) for t in records_as_tuples]
    save_results(db_name, query, list(set(rearranged_tuples)))
    
#preprocessing()
#create_result_table('results.db')
#ilis = get_ilis_with_conf_1_0()
#get_llm_synset_target(ilis)

#preprocessing_with_de_inferred()
#create_result_table('results_de.db')
#create_result_table('results_af.db')
#copy_results('results.db', 'results_de.db', 'de')
#copy_results('results.db', 'results_af.db', 'af')

#preprocessing()
# create_result_table('results_de.db')
# create_result_table('results_af.db')
# trans_lemma_dict_de = get_lemma_trans('ili_lemma_de.txt')
# trans_lemma_dict_af = get_lemma_trans('ili_lemma_af.txt')
# populate_eval_table('results_de.db', trans_lemma_dict_de, 'de')
#populate_eval_table('results_af.db', trans_lemma_dict_af, 'af')


#re_eval_af('results_af.db')

#preprocessing()
#mismatch_eval_de('results_de.db')

#import_eval('results_de.db', 'Auswertung1_Joerg.csv', EVAL_DE_MISMATCH_TABLE_UPDATE)
#import_eval('results_de.db', 'Auswertung1_Joerg_llm_judge.csv', LLM_AS_A_JUDGE_UPDATE)

#preprocessing()
#ilis = get_ilis_with_conf_1_0()
#llm_as_a_judge(ilis)

#import_eval('results_af.db', 'Auswertung_af_overlap_llm_judge.csv', LLM_AS_A_JUDGE_UPDATE)
#import_eval('results_af.db', 'Auswertung_af_llm_judge.csv', LLM_AS_A_JUDGE_UPDATE)


preprocessing()