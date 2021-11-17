import pandas as pd
import re
import json
import xml.etree.ElementTree as ET
from itertools import chain 

dataset = []
dataset_invalid = []
dataset_fixed = []
au_count = 0
for j in range(1,403):
    f = open('../../ceph_data/input/corpus-ukp-argument-annotated-essays-v2/uima/webis-format/essay'+str(j).zfill(3)+'.xmi', "r")
    txt = f.read()
    f.close()

    txt = re.sub(r'(argumentation)+\:|(xmi)+\:|(cas)+\:', '', txt)


    root = ET.fromstring(txt)

    topic = root.find('MetadataAAE').get('topic')
    essay = root.find('Sofa').get('sofaString')


    claims_and_premises = {}
    for adu in root.findall('ArgumentativeDiscourseUnit'):
        if 'major' not in adu.get('unitType'):
            claims_and_premises[adu.get('id')] = re.sub(r'\n', ' ', essay[int(adu.get('begin')):int(adu.get('end'))])

    claim_premise_dict = {}
    for arg in root.findall('Argument'):
        if arg.get('conclusion') in claims_and_premises and arg.get('premises') in claims_and_premises:
            if arg.get('conclusion') not in claim_premise_dict:
                claim_premise_dict[arg.get('conclusion')] = {'index': 'essay'+str(j).zfill(3), 'topic': topic, 'para_text': re.sub(r'\n', ' ', essay), 'premises': [claims_and_premises[arg.get('premises')]], 'conclusion': claims_and_premises[arg.get('conclusion')]}      
            else:
                claim_premise_dict[arg.get('conclusion')]['premises'].append(claims_and_premises[arg.get('premises')])

    # add score
    arg_score_dict = {}
    au_count += len(root.findall('LocalSufficiency'))
    for score in root.findall('LocalSufficiency'):
        arg = re.sub(r'\n', ' ', essay[int(score.get('begin')):int(score.get('end'))])
        arg_score_dict[score.get('id')] = {'au': arg, 'majority': score.get('majority'), 'mean': score.get('mean')}


    # remove NA
    for key1, value1 in claim_premise_dict.items():
        num_paragraph = 1
        for key2, value2 in arg_score_dict.items():
            if value1['conclusion'] in value2['au']:
                claim_premise_dict[key1]['local_sufficency'] = value2['majority']
                claim_premise_dict[key1]['au'] = value2['au']
                claim_premise_dict[key1]['index'] += '_'+str(num_paragraph)
            num_paragraph += 1
        if 'au' not in  claim_premise_dict[key1]:
            claim_premise_dict[key1]['au'] = 'NA'

    # fixing intersections
    conclusions = []
    premises = []
    for arg in root.findall('Argument'):
        if arg.get('conclusion') in claims_and_premises and arg.get('premises') in claims_and_premises:
            conclusions.append(arg.get('conclusion'))
            premises.append(arg.get('premises'))
    intersection = list(set(conclusions) & set(premises))
    for arg in root.findall('Argument'):
        if arg.get('conclusion') in claims_and_premises and arg.get('premises') in claims_and_premises:
            if arg.get('premises') in intersection:
                for premise in claim_premise_dict[arg.get('premises')]['premises']:
                    claim_premise_dict[arg.get('conclusion')]['premises'].append(premise)
    claim_premise_dic_fixed = {}
    for value in intersection:
        claim_premise_dic_fixed[value] = claim_premise_dict[value]
        del claim_premise_dict[value]



    # adding corresponding major claims
    claims_and_premises = {}
    for adu in root.findall('ArgumentativeDiscourseUnit'):
        if 'major' in adu.get('unitType'):
            claims_and_premises[adu.get('id')] = re.sub(r'\n', ' ', essay[int(adu.get('begin')):int(adu.get('end'))])
    for arg in root.findall('Argument'):
        if arg.get('conclusion') in claims_and_premises:
            if arg.get('premises') in claim_premise_dict:
                claim_premise_dict[arg.get('premises')]['major_claim'] = claims_and_premises[arg.get('conclusion')]
    
    delete_na = []
    claim_premise_dic_invalid = {}
    for key, value in claim_premise_dict.items():
        if value['au'] == 'NA':
            claim_premise_dic_invalid[key] = claim_premise_dict[key]
            claim_premise_dic_invalid[key]['removal_reason'] = 'NA'
            delete_na.append(key)

    for key in delete_na:
        del claim_premise_dict[key]

    rev_dict = {} 
    for key, value in claim_premise_dict.items(): 
        rev_dict.setdefault(value['au'], set()).add(key) 
    
    result = set(chain.from_iterable( 
        values for key, values in rev_dict.items() 
        if len(values) > 1)) 

    for value in result:
        claim_premise_dic_invalid[value] = claim_premise_dict[value]
        claim_premise_dic_invalid[value]['removal_reason'] = 'multi'
        del claim_premise_dict[value]

    # sort premises by appearance
    for arg in claim_premise_dict.values():
        arg['premises'].sort(key=lambda c: arg['para_text'].index(c))

    assert(len([value for key, value in claim_premise_dict.items()]) <= len(root.findall('LocalSufficiency')))

    if len([value for key, value in claim_premise_dict.items()]) < len(root.findall('LocalSufficiency')):
        print(j)
        print(len([value for key, value in claim_premise_dict.items()])-len(root.findall('LocalSufficiency')))
        print('-------------')
    dataset+=[value for key, value in claim_premise_dict.items()]
    dataset_fixed+=[value for key, value in claim_premise_dic_fixed.items()]
    dataset_invalid+=[value for key, value in claim_premise_dic_invalid.items()]

print(len(dataset_fixed))
print(len(dataset_invalid))
print(len(dataset))
print(au_count)
with open('../../ceph_data/intermediate/corpus-ukp-argument-annotated-essays-v2/aae3.json', 'w') as f:
    json.dump(dataset, f)

with open('../../ceph_data/intermediate/corpus-ukp-argument-annotated-essays-v2/aae3_fixed.json', 'w') as f:
    json.dump(dataset_fixed, f)

with open('../../ceph_data/intermediate/corpus-ukp-argument-annotated-essays-v2/aae3_invalid.json', 'w') as f:
    json.dump(dataset_invalid, f)