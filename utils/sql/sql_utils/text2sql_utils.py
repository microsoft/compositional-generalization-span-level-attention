import collections
import re
from typing import List, Tuple, Iterator

from utils.sql.sql_utils.encoder_input_canonicalizer import process_sentence as preprocess_text

table_alias_re = re.compile(r'[A-Z_]+ AS [A-Z_]+alias[0-9]([\s],)?')


CANONICAL_VARIABLES = {
    'aircraft_code0',
    'airline_code0',
    'airline_code1',
    'airline_name0',
    'airport_code0',
    'airport_code1',
    'airport_name0',
    'arrival_time0',
    'arrival_time1',
    'arrival_time2',
    'basic_type0',
    'booking_class0',
    'booking_class1',
    'city_name0',
    'city_name1',
    'city_name2',
    'city_name3',
    'class_type0',
    'class_type1',
    'connections0',
    'country_name0',
    'day_name0',
    'day_name1',
    'day_name2',
    'day_name3',
    'day_name4',
    'day_number0',
    'day_number1',
    'days_code0',
    'departure_time0',
    'departure_time1',
    'departure_time2',
    'departure_time3',
    'discounted0',
    'economy0',
    'fare_basis_code0',
    'fare_basis_code1',
    'flight_days0',
    'flight_number0',
    'flight_number1',
    'manufacturer0',
    'meal_code0',
    'meal_description0',
    'month_number0',
    'one_direction_cost0',
    'propulsion0',
    'round_trip_cost0',
    'round_trip_required0',
    'state_code0',
    'state_name0',
    'state_name1',
    'state_name2',
    'stops0',
    'time_elapsed0',
    'transport_type0',
    'transport_type1',
    'year0',
    'year1'
}


def canonicalize_sql_for_alignment(sql_tokens: List[str]) -> Tuple[str, dict]:
    # sql = table_alias_re.sub('', sql)
    # sql = re.sub(r'\s+', ' ', sql)
    canonical_sql, mapping = shorten_sql_tokens(sql_tokens)

    return canonical_sql, mapping


def shorten_sql_tokens(s_toks):
    """
    filter tokens, return the map
    :param s_toks: tokenized sql string
    :return: filtered tokens, mapping to s_toks
    """
    new_toks = []
    mapping = []
    i = 0
    while i < len(s_toks) - 1: # remove all ;
        # remove all "FROM" clauses with commas
        if i < len(s_toks) - 3 \
                and re.sub(r'[A-Z_]+ AS [A-Z_]+alias[0-9]', '',  ' '.join(s_toks[i:i+3])) == '':
            i += 3
        elif s_toks[i] in {',', '"'}:
            i += 1
        ## remove SELECT FROM WHERE
        # elif s_toks[i] in ('SELECT', 'FROM', ',', '"'):
        #     i += 1
        # concatenate table . column
        elif i < len(s_toks) - 3 \
                and re.sub(r'[A-Z_]+alias[0-9] \. [A-Z_]+', '',  ' '.join(s_toks[i:i+3])) == '':
            new_toks.extend([''.join(s_toks[i:i+3])])
            mapping.append((i, i+2))
            i += 3
        # # concatenate GROUPBY AND ORDERBY
        # elif i < len(s_toks) - 2 \
        #         and ' '.join(s_toks[i:i + 2]) in ("GROUP BY", "ORDER BY"):
        #     new_toks.extend([''.join(s_toks[i:i + 2])])
        #     mapping.append((i, i + 1))
        #     i += 2
        # # remove LIMIT
        # elif i < len(s_toks) - 3 \
        #         and re.sub(r'LIMIT [0-9]', '',  ' '.join(s_toks[i:i+2])) == '':
        #     i += 2
        else:
            new_toks.append(s_toks[i])
            mapping.append((i,i))
            i += 1
    assert all([isinstance(e, tuple) for e in mapping]), f"non tuple in the mapping file! {mapping}"
    return new_toks, mapping


class SqlTokenizer:
    """
    splits columns and table names, splits quotes from values,
    splits parenthesis from sql tokens
    """

    @staticmethod
    def clean(text):
        """
        fix aliased tables and columns to be the same token, sql values as
        :param text:
        :return:
        """
        # if not text.startswith('SELECT'):
        #     return text
        text = re.sub(r'([^ ]+alias[0-9])\.([^ ]+)', r'\g<1> . \g<2>', text)
        text = text.replace('"', ' " ').replace('(', ' ( ').replace(')', ' ) ')
        # I added this last replace() change after running seq2seq experiments
        # text = text.replace(',', ' ,')
        text = re.sub(r'[ ]+', ' ', text)
        return text

    def tokenize(self, text: str) -> List[str]:
        return [t for t in self.clean(text).split()]


def process_sql_data_standard(
    data,
    use_linked,
    use_all_sql,
    use_all_queries,
    output_spans=False
) -> Iterator[Tuple[str, str]]:
    """
    Reads pairs of (sentence, sql) from data. gives different results than "process_sql_data" since the
    "seen_sentences" set is initialized once for all the sentences in data (and not for every entry...)
    """
    text2sql_pairs = collections.OrderedDict()  # set of tuples (question, sql query)
    linked_text2sql = collections.OrderedDict()  # set of tuples (question with variables, sql with variables)
    for entry in data:
        all_sql = [entry["sql"][0]] if not use_all_sql else entry["sql"]
        for sql in all_sql:
            seen_texts = set()  # set of all seen questions
            seen_linked_texts = set()  # set of all seen texts with variables (=linked entities)
            for utt in entry["sentences"]:
                if utt['question-split'] == 'exclude':
                    continue
                built_sql = sql
                text = utt["text"]
                built_text = text
                for k, v in utt["variables"].items():
                    built_sql = built_sql.replace(k, v)
                    built_text = built_text.replace(k, v)
                built_text = preprocess_text(built_text)
                text = preprocess_text(text)
                # fix `` to " back
                built_text = built_text.replace('``', '\"')
                text = text.replace('``', '\"')
                # convert from List[Tuple[int, int]] to string
                spans = ' '.join([f"{s[0]}-{s[1]}" for s in utt['constituency_parser_spans']])
                if use_all_queries:
                    text2sql_pairs[(built_text, built_sql)] = True
                    linked_text2sql[(text, sql, spans)] = True
                else:
                    if built_text not in seen_texts:  # don't add two texts with same sql
                        seen_texts.add(built_text)
                        text2sql_pairs[(built_text, built_sql)] = True
                    if text not in seen_linked_texts:  # don't add two texts with different sql
                        seen_linked_texts.add(text)
                        linked_text2sql[(text, sql)] = True

    if not use_linked:
        for pair in text2sql_pairs.keys():
            yield pair
    else:
        for pair in linked_text2sql.keys():
            if output_spans:
                # convert back to List[Tuple[int, int]]
                spans = [(int(span.split('-')[0]), int(span.split('-')[1])) for span in pair[2].split()]
                yield pair[0], pair[1], spans
            else:
                yield pair[0], pair[1]
