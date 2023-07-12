import os
import json
import ast
import re
import numpy as np
import pickle

import functions

np.random.seed(42)

# Parameters
BASEDIR = 'data'
LINES = 'all'
FILES = ['2021-05-01', '2021-06-28', '2022-01-03']
FIELDS = ['profilename', 'testcase', 'profilename_testcase']

json_reference_tree = {}
with open('data/tree_structure_complete.json', 'r') as reference_tree_file:
    json_reference_tree = json.loads(reference_tree_file.read())

system_commands = ['mm', 'ls', 'grep', 'test', 'exit', 'admin', 'timeout', 'y', 'maintenance', 'q', 'co', 'sh', 'cat',
                   'ping', 'yes']

for file in FILES:
    for field in FIELDS:

        filename = os.path.join(BASEDIR, file)
        log_file = open(filename, 'r')

        documents = {}
        n = 0
        for line in log_file:
            parsed_line = ast.literal_eval(line)

            # print(n, parsed_line['cmd'])
            # selection of the field (profile_name, tc_case, both)
            if field == 'profilename':
                feature = parsed_line['profile_name']
            elif field == 'testcase':
                feature = parsed_line['tc_name']
            else:
                feature = parsed_line['profile_name'] + '_' + parsed_line['profile_name']

            # a dictionary is created containing the commands for each field
            if not feature in documents:
                documents[feature] = []
            split_command = re.split(r'\s+', parsed_line['cmd'])
            if not split_command[0] in system_commands:  # Is not a system commmand
                parsed_log_command = functions.log_parser(json_reference_tree, parsed_line['cmd'])
                documents[feature].append(parsed_log_command)

            n += 1

            if LINES != 'all':
                if n > LINES:
                    break

        # the dictionary is saved
        path = 'workspace'
        if not os.path.exists(path):
            os.makedirs(path)
            os.makedirs('{}/results'.format(path))
            print('workspace folder created')
        filename = '{}/documents_{}_{}.pkl'.format(path, field, file)
        with open(filename, 'wb') as f:
            pickle.dump(documents, f)
