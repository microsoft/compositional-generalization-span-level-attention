# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Lint as: python3
"""Library for simplifying SPARQL queries."""

import collections
import enum
import re

from absl import logging


class SimplifyFunction(enum.Enum):
  DO_NOTHING = 0
  GROUP_SUBJECTS = 1
  GROUP_SUBJECTS_AND_OBJECTS = 2
  GROUP_SUBJECTS_AND_OBJECTS_AND_SORT = 3


def get_relation(clause):
  clause = clause.strip()
  if clause.startswith('FILTER'):
    matches = re.match(r'FILTER \( (.*) != (.*) \)', clause)
    assert matches, f'Invalid FILTER clause: {clause}'
    subj, rel, obj = matches.group(1), 'not', matches.group(2)
  else:
    subj, rel, obj = clause.split(' ')
  return (subj, rel, obj)


def group_subjects(prefix, clauses):
  """This is the implementation of function F1."""
  edges = collections.defaultdict(list)
  for clause in clauses:
    subj, rel, obj = get_relation(clause)
    edges[subj].append(f'{rel} {obj}')

  output = f'{prefix} '
  for subj, rels in edges.items():
    output += subj + ' { ' + ' . '.join(rels) + ' } '
  return output.strip()


def group_subjects_and_objects(prefix, clauses, sort=False, return_metadata: bool = False):
  """This is the implementation of F2 (sort=False) and F3 (sort=True)."""

  # edges = collections.defaultdict(lambda: collections.defaultdict(list))
  edges = collections.OrderedDict()

  for clause in clauses:
    subj, rel, obj = get_relation(clause)
    (
      edges
      .setdefault(subj, collections.OrderedDict())
      .setdefault(rel, [])
      .append(obj)
    )
    # edges[subj][rel].append(obj)

  char_offest = 0
  output = prefix + ' '
  char_offest += len(output)

  subjects = edges.keys()

  if sort:
    subjects = sorted(subjects)

  meta_info = {
    'prefix_char_offset': (0, char_offest),
    'subjects': []
  }

  for subj in subjects:
    subject_prefix = subj + ' { '
    output += subject_prefix

    subj_meta_entry = {
      'subject': subj,
      'char_offset': (char_offest, char_offest + len(subj)),
      'char_offset_with_trivia': (char_offest, char_offest + len(subject_prefix)),
      'relations': []
    }

    char_offest += len(subject_prefix)

    targets = edges[subj]
    rels = targets.keys()

    if sort:
      rels = sorted(rels)

    for rel in rels:
      rel_string = rel + ' { ' + ' '.join(targets[rel]) + ' } '
      output += rel_string

      subj_meta_entry['relations'].append({
        'relation': rel,
        'char_offset': (char_offest, char_offest + len(rel)),
        'char_offset_with_trivia': (char_offest, char_offest + len(rel + ' { ')),
        'objects': []
      })
      char_offest += len(rel + ' { ')

      objects_meta = []
      for obj in targets[rel]:
        objects_meta.append({
          'object': obj,
          'char_offset': (char_offest, char_offest + len(obj))
        })
        char_offest += len(obj + ' ')

      subj_meta_entry['relations'][-1]['objects'] = objects_meta

      char_offest += len('} ')

    meta_info['subjects'].append(subj_meta_entry)

    output += '} '
    char_offest += len('} ')

  assert len(output) == char_offest

  if return_metadata:
    return output.strip(), meta_info
  else:
    return output.strip()


def rewrite(
  query,
  simplify_function = SimplifyFunction.DO_NOTHING,
  return_metadata: bool = False
):
  """Rewrites SPARQL according to the given simplifying function."""
  if simplify_function == SimplifyFunction.DO_NOTHING:
    return query
  logging.info('Rewriting %s', query)
  matches = re.match('(.*){(.*)}', query)
  assert matches, f'Invalid SPARQL: {query}'
  prefix, clauses = matches.group(1), matches.group(2).split(' . ')
  # Prefix is either 'SELECT count' or 'SELECT DISTINCT'
  prefix = prefix.split()[1].upper()

  return {
      SimplifyFunction.GROUP_SUBJECTS:
          group_subjects(prefix, clauses),
      SimplifyFunction.GROUP_SUBJECTS_AND_OBJECTS:
          group_subjects_and_objects(prefix, clauses),
      SimplifyFunction.GROUP_SUBJECTS_AND_OBJECTS:
          group_subjects_and_objects(prefix, clauses, return_metadata=return_metadata),
      SimplifyFunction.GROUP_SUBJECTS_AND_OBJECTS_AND_SORT:
          group_subjects_and_objects(prefix, clauses, sort=True, return_metadata=return_metadata),
  }[simplify_function]


if __name__ == '__main__':
  #query = """SELECT count ( * ) WHERE { ?x0 film.editor.film M0 . M1 people.person.spouse_s/ns:people.marriage.spouse|ns:fictional_universe.fictional_character.married_to/ns:fictional_universe.marriage_of_fictional_characters.spouses ?x0 . M2 people.person.spouse_s/ns:people.marriage.spouse|ns:fictional_universe.fictional_character.married_to/ns:fictional_universe.marriage_of_fictional_characters.spouses ?x0 . M3 people.person.spouse_s/ns:people.marriage.spouse|ns:fictional_universe.fictional_character.married_to/ns:fictional_universe.marriage_of_fictional_characters.spouses ?x0 . M4 people.person.spouse_s/ns:people.marriage.spouse|ns:fictional_universe.fictional_character.married_to/ns:fictional_universe.marriage_of_fictional_characters.spouses ?x0 . M5 people.person.spouse_s/ns:people.marriage.spouse|ns:fictional_universe.fictional_character.married_to/ns:fictional_universe.marriage_of_fictional_characters.spouses ?x0 }"""
  query = """SELECT count ( * ) WHERE { ?x0 a film.editor . M1 film.film.directed_by ?x0 . M1 film.film.written_by ?x0 . M2 film.film.directed_by ?x0 . M2 film.film.written_by ?x0 . M3 film.film.directed_by ?x0 . M3 film.film.written_by ?x0 . M4 film.film.directed_by ?x0 . M4 film.film.written_by ?x0 . M5 film.film.directed_by ?x0 . M5 film.film.written_by ?x0 }"""
  rewritten_query = rewrite(query, SimplifyFunction.GROUP_SUBJECTS_AND_OBJECTS_AND_SORT)
  print(rewritten_query)
