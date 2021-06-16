import argparse
import random
from aser.database.db_API import KG_Connection
from aser.database.db_API import generate_event_id, generate_relation_id, generate_id

def test_DB(kg_conn):
    print ('BEGIN TEST...')
    print ('SIZE:', 'eventualities: ', len(kg_conn.event_id_set), 'relations:', len(kg_conn.relation_id_set))
    print (list(zip(kg_conn.event_columns, kg_conn.event_column_types)))
    print (list(zip(kg_conn.relation_columns, kg_conn.relation_column_types)))
    print (kg_conn.get_exact_match_event(generate_id('i learn python')))
    print (kg_conn.get_exact_match_relation([generate_id('i be tired'), generate_id('i sleep')]))
    print ('TEST DONE!')

def find_pattern(relation, sent1, sent2):
    result = []
    best_key = None
    best_value = 0.0
    for key in relation:
        if key == '_id' or key == 'event1_id' or key == 'event2_id' or key == 'Co_Occurrence':
            continue
        if relation[key] > best_value:
            best_key = key
            best_value = relation[key]

    if best_key is not None and best_value > 0.0:
        if best_key == 'Precedence':
            r = [sent1 + ' before ' + sent2 + '.',
                sent1 + ', then ' + sent2 + '.',
                sent1 + ' till ' + sent2 + '.',
                sent1 + ' until ' + sent2 + '.']
            result.append(random.choice(r))
        elif best_key == 'Succession':
            r = [sent1 + ' after ' + sent2 + '.',
                sent1 + ' once ' + sent2 + '.']
            result.append(random.choice(r))
        elif best_key == 'Synchronous':
            r = [sent1 + ', meanwhile ' + sent2 + '.',
                sent1 + ' meantime ' + sent2 + '.',
                sent1 + ', at the same time ' + sent2 + '.']
            result.append(random.choice(r))
        elif best_key == 'Reason':
            result.append(sent1 + ', because ' + sent2 + '.')
        elif best_key == 'Result':
            r = [sent1 + ', so ' + sent2 + '.',
                sent1 + ', thus ' + sent2 + '.',
                sent1 + ', therefore ' + sent2 + '.',
                sent1 + ', so that ' + sent2 + '.']
            result.append(random.choice(r))
        elif best_key == 'Condition':
            r = [sent1 + ', if ' + sent2 + '.',
                    sent1 + ', as long as ' + sent2 + '.']
            result.append(random.choice(r))
        elif best_key == 'Contrast':
            r = [sent1 + ', but ' + sent2 + '.',
                sent1 + ', however ' + sent2 + '.',
                sent1 + ', by contrast ' + sent2 + '.',
                sent1 + ', in contrast ' + sent2 + '.',
                sent1 + ', on the other hand ' + sent2 + '.',
                sent1 + ', on the contrary ' + sent2 + '.']
            result.append(random.choice(r))
        elif best_key == 'Concession':
            result.append(sent1 + ', although ' + sent2 + '.')
        elif best_key == 'Conjunction':
            r = [sent1 + ' and ' + sent2 + '.',
                 sent1 + ', also ' + sent2 + '.']
            result.append(random.choice(r))
        elif best_key == 'Instantiation':
            r = [sent1 + ', for example ' + sent2 + '.',
                sent1 + ', for instance ' + sent2 + '.']
            result.append(random.choice(r))
        elif best_key == 'Restatement':
            result.append(sent1 + ', in other words ' + sent2 + '.')
        elif best_key == 'Alternative':
            r = [sent1 + ' or ' + sent2 + '.',
                sent1 + ', unless ' + sent2 + '.',
                sent1 + ', as an alternative ' + sent2 + '.',
                sent1 + ', otherwise ' + sent2 + '.']
            result.append(random.choice(r))
        elif best_key == 'ChosenAlternative':
            result.append(sent1 + ', ' + sent2 + ' instead.')
        elif best_key == 'Exception':
            result.append(sent1 + ', except ' + sent2 + '.')
    return result

def convert1(kg_conn, verb_file_path, output_file_path):
	# the reimplement of the data generation approach used by ASER (Zhang et. al.)
    print ('BEGIN CONVERT...')

    verb_set = []

    with open(verb_file_path) as f:
        lines = list(f)
        for id_x,(sent,pronoun,candidates,candidate_a,_) in enumerate(zip(lines[0::5],lines[1::5],lines[2::5],lines[3::5],lines[4::5])):
            A_id = sent.index('A')
            B_id = sent.index('B')
            verb = sent[A_id+1:B_id].strip()
            if verb not in verb_set:
                verb_set.append(verb)

    print (verb_set)

    with open(output_file_path, 'w') as f:
        for relation_id in kg_conn.relation_id_set:
            relation = kg_conn.get_exact_match_relation(relation_id)
            event1_id = kg_conn.get_exact_match_event(relation['event1_id'])
            event2_id = kg_conn.get_exact_match_event(relation['event2_id'])
            event1 = kg_conn.get_exact_match_event(event1_id)
            event2 = kg_conn.get_exact_match_event(event2_id)

            sent1 = event1['words'].strip()
            sent1 = sent1.split(event1['verbs'])
            if event1['pattern'] not in ['s-v-o'] or event2['pattern'] not in ['s-be-a'] or event1['verbs'] not in verb_set:
                continue
            if len(sent1) != 2 or sent1[0] == '' or sent1[1] == '':
                continue

            sent2 = event2['words'].strip().split()
            tmp_sent2 = []
            for word in sent2:
                if word == 'be':
                    tmp_sent2.append('is')
                else:
                    tmp_sent2.append(word)

            if relation['Reason'] + relation['Result'] + relation['Condition'] == 0.0:
                continue

            part_A = sent1[0].strip().split()
            part_B = sent1[1].strip().split()
            
            pronoun = 'he'
            candidates = 'A,B'

            for word in part_A:
                sent2 = tmp_sent2
                true_candidate = None
                if word in sent2:
                    sent2[sent2.index(word)] = 'he'
                    question = 'A ' + event1['verbs'] + ' B, ' + ' '.join(sent2) + '.'
                    true_candidate = 'A'
                    f.write('{}\n{}\n{}\n{}\n\n'.format(question, pronoun, candidates, true_candidate))
            
            for word in part_B:
                sent2 = tmp_sent2
                true_candidate = None
                if word in sent2:
                    sent2[sent2.index(word)] = 'he'
                    question = 'A ' + event1['verbs'] + ' B, ' + ' '.join(sent2) + '.'
                    true_candidate = 'B'
                    f.write('{}\n{}\n{}\n{}\n\n'.format(question, pronoun, candidates, true_candidate))
                
    print ('CONVERT DONE!')

def convert2(kg_conn, output_file_path, enrich_file_path):
	# the data generation approach we used in the paper
    import stanza
    import numpy as np
    nlp = stanza.Pipeline('en', processors='tokenize,mwt,pos')
    print ('BEGIN CONVERT...')
    with open(output_file_path, 'w') as f:
        enrich_file = open(enrich_file_path, 'w')
        for relation_id in kg_conn.relation_id_set:
            relation = kg_conn.get_exact_match_relation(relation_id)
            event1_id = kg_conn.get_exact_match_event(relation['event1_id'])
            event2_id = kg_conn.get_exact_match_event(relation['event2_id'])
            event1 = kg_conn.get_exact_match_event(event1_id)
            event2 = kg_conn.get_exact_match_event(event2_id)
            sent1 = event1['words'].strip()
            sent2 = event2['words'].strip()

            if relation['Precedence'] + relation['Succession'] + relation['Synchronous'] + relation['Reason'] + relation['Result'] + relation['Condition'] + relation['Contrast'] + relation['Concession'] + relation['Conjunction'] + relation['Instantiation'] + relation['Restatement'] + relation['ChosenAlternative'] + relation['Alternative'] + relation['Exception'] == 0.0:
                if relation['Co_Occurrence'] > 0:
                    enrich_file.write('{}\t{}\t{}\t{}\n'.format(sent1, sent2, 'Precedence', relation['Co_Occurrence']))
            else:
	            result1 = nlp(sent1)
	            sentence1 = result1.sentences[0]
	            s1 = []
	            for word in sentence1.words:
	                if word.upos == 'NOUN' or word.upos == 'PROPN' or word.upos == 'PRON':
	                    s1.append(1)
	                else:
	                    s1.append(0)
	            result2 = nlp(sent2)
	            sentence2 = result2.sentences[0]
	            s2 = []
	            for word in sentence2.words:
	                if word.upos == 'NOUN' or word.upos == 'PROPN' or word.upos == 'PRON':
	                    s2.append(1)
	                else:
	                    s2.append(0)
	            
	            if np.array(s1).sum() < 2 or np.array(s2).sum() < 1:
	                continue

	            sentence1 = [word.text for word in sentence1.words]
	            sentence2 = [word.text for word in sentence2.words]

	            candidates_list = []
	            for num1, tag1 in enumerate(s1):
	                if tag1 == 1:
	                    candidates_list.append(num1)

	            for candidate_a in candidates_list:
	                text = sentence1[candidate_a]
	                for num2, tag2 in enumerate(s2):
	                    if sentence2[num2] == text and s2[num2] == 1:
	                        for candidate_b in candidates_list:
	                            if candidate_b != candidate_a:
	                                part_a = sentence1.copy()
	                                part_b = sentence2.copy()
	                                part_b[num2] = '[MASK]'

	                                questions = find_pattern(relation, ' '.join(part_a), ' '.join(part_b))
	                                for question in questions:
	                                    candidates = sentence1[candidate_a]+','+sentence1[candidate_b]
	                                    true_candidate = sentence1[candidate_a]
	                                    f.write('{}\n{}\n{}\n{}\n\n'.format(question, '[MASK]', candidates, true_candidate))
        enrich_file.close()
    print ('CONVERT DONE!')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-db', '--DB_PATH', help='PATH of Database.', default='../ASER-DB/core/KG_v0.1.0.db')
    parser.add_argument('-mode', '--DB_MODE', help='MODE of Database.', default='cache')
    parser.add_argument('-w', '--VERB_FILE_PATH', help='PATH of WSC273 VERB File.', default=None)
    parser.add_argument('-o', '--OUTPUT_FILE_PATH', help='PATH of Output File.', default='../data/ASER-convert/ASER-output-core') # existing eventuality relation mentions after removing connectives
    parser.add_argument('-e', '--ENRICH_FILE_PATH', help='PATH of Enrich File.', default='../data/ASER-convert/ASER-enrich-core') # eventualities co-occurrence in the same sentence without explicit connectives;
    args = parser.parse_args()

    kg_conn = KG_Connection(db_path=args.DB_PATH, mode=args.DB_MODE)
    test_DB(kg_conn)
    random.seed(42)
    # convert1(kg_conn, args.VERB_FILE_PATH, args.OUTPUT_FILE_PATH)
    convert2(kg_conn, args.OUTPUT_FILE_PATH, args.ENRICH_FILE_PATH)