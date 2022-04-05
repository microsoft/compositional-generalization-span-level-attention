# -*- coding: utf-8 -*-
# @Author: ZwEin
# @Date:   2016-07-19 19:16:31
# @Last Modified by:   ZwEin
# @Last Modified time: 2016-11-21 15:29:27


"""
Usage Example

# parse string
python sqparser.py -s "<SPARQL_QUERIES_STRING>"

# parse json file
python sqparser.py -i <INPUT_FILE_PATH> -o <OUTPUT_FILE_PATH> -t <TRUE|FALSE> -c <'WHERE' by default>

# parse sparql-queries.json that contains title, and only extract WHERE component
python sqparser.py -i tests/data/sparql-queries.json -o test.json -t True -c "WHERE"

# parse sparql-queries.json that doesn't contain title, and only extract WHERE component
python sqparser.py -i tests/data/sparql-queries-without-title.json -o test.json -t False -c "WHERE"

"""

import re
import json

######################################################################
#   Constant
######################################################################

# schema
SQ_SCHEMA_PAYLOAD = 'payload'

# keyword
SQ_KEYWORD_SELECT = 'SELECT'
SQ_KEYWORD_PREFIX = 'PREFIX'
SQ_KEYWORD_WHERE = 'WHERE'
SQ_KEYWORD_ORDER = 'ORDER'
SQ_KEYWORD_GROUP = 'GROUP'
SQ_KEYWORD_LIMIT = 'LIMIT'
SQ_KEYWORD_FILTER = 'FILTER'
SQ_KEYWORD_OPTIONAL = 'OPTIONAL'
SQ_KEYWORD_BIND = 'BIND'

# operator
SQ_OPERATOR_OR = '||'
SQ_OPERATOR_AND = '&&'

SQ_OUTER_KEYWORDS = ['SELECT', 'PREFIX', 'WHERE', 'ORDER', 'GROUP', 'LIMIT']
SQ_INNER_KEYWORDS = ['FILTER', 'OPTIONAL', 'BIND']

SQ_OUTER_OPERATOR = [SQ_OPERATOR_OR, SQ_OPERATOR_AND]
SQ_INNER_OPERATOR = ['!=', '<=', '>=', '<', '>', '==', '=']  # keep in this order

SQ_OPERATOR_MAPPING = {
    SQ_OPERATOR_OR: 'OR',
    SQ_OPERATOR_AND: 'AND'
}

# SQ_KEYWORDS = ['SELECT','CONSTRUCT','DESCRIBE','ASK','BASE','PREFIX','LIMIT','OFFSET','DISTINCT','REDUCED','ORDER','BY','ASC','DESC','FROM','NAMED','WHERE','GRAPH','OPTIONAL','UNION','FILTER']

# SQ_FUNCTIONS = ['STR','LANGMATCHES','LANG','DATATYPE','BOUND','sameTerm','isIRI','isURI','isBLANK','isLITERAL','REGEX']

# extraction names
SQ_EXT_TYPE = 'type'
SQ_EXT_VARIABLE = 'variable'
SQ_EXT_CLAUSES = 'clauses'
SQ_EXT_PREDICATE = 'predicate'
SQ_EXT_CONSTAINT = 'constraint'
SQ_EXT_FILTERS = 'filters'
SQ_EXT_OPTIONAL_FLAG = 'isOptional'
SQ_EXT_OPERATOR = 'operator'
SQ_EXT_GOL = 'group-by'
SQ_EXT_GROUP_VARIABLE = 'group-variable'
SQ_EXT_ORDER_VARIABLE = 'order-variable'
SQ_EXT_SORTED_ORDER = 'sorted-order'
SQ_EXT_LIMIT = 'limit'

# functions
SQ_FUNCTION_BIND = 'bind'
SQ_FUNCTION_BOUND = 'bound'
SQ_FUNCTION_ASC = 'asc'
SQ_FUNCTION_DESC = 'desc'
SQ_FUNCTION_COUNT = 'count'
SQ_FUNCTION_AVG = 'avg'
SQ_FUNCTION_MIN = 'min'
SQ_FUNCTION_MAX = 'max'
SQ_FUNCTION_GROUP_CONCAT = 'group_concat'
SQ_FUNCTION_FILTER_CONTAINS = 'contains'
SQ_FUNCTION_FILTER_LCASE = 'lcase'

SQ_FUNCTIONS = [  # modify SQ_FUNCTION_FUNC also, if update
    SQ_FUNCTION_BIND,
    SQ_FUNCTION_BOUND,
    SQ_FUNCTION_ASC,
    SQ_FUNCTION_DESC,
    SQ_FUNCTION_COUNT,
    SQ_FUNCTION_AVG,
    SQ_FUNCTION_MIN,
    SQ_FUNCTION_MAX,
    SQ_FUNCTION_GROUP_CONCAT,
    SQ_FUNCTION_FILTER_CONTAINS,
    SQ_FUNCTION_FILTER_LCASE
]

######################################################################
#   Regular Expression
######################################################################

re_continues_digits = re.compile(r'\d+')

re_brackets_most_b = re.compile(r'(?<={).*(?=})')
re_brackets_least_b = re.compile(r'(?<={).*?(?=})')
re_brackets_most_m = re.compile(r'(?<=\[).*(?=\])')
re_brackets_least_m = re.compile(r'(?<=\[).*?(?=\])')
re_brackets_most_s = re.compile(r'(?<=\().*(?=\))')
re_brackets_least_s = re.compile(r'(?<=\().*?(?=\))')

re_variable = re.compile(r'(?<=[\(\b\s])\?[\-_a-zA-Z]+(?=[\)\b\s\\Z]|$)')

# keyword
reg_outer = r'(?:' + r'|'.join(SQ_OUTER_KEYWORDS) + r').*?(?=' + r'|'.join(SQ_OUTER_KEYWORDS) + r'|\s*$)'
re_outer = re.compile(reg_outer)
reg_inner = r'(?:' + r'|'.join(SQ_INNER_KEYWORDS) + r').*?(?=' + r'|'.join(SQ_INNER_KEYWORDS) + r'|\s*$)'
re_inner = re.compile(reg_inner)

re_keyword = re.compile(r'^[a-zA-Z]+\b')

# operator
reg_outer_operator = r'(?:' + r'|'.join(SQ_OUTER_OPERATOR) + r').*?(?=' + r'|'.join(SQ_OUTER_OPERATOR) + r'|\s*$)'
re_outer_operator = re.compile(reg_outer_operator)
re_outer_operator_split = re.compile(r'[' + r''.join(SQ_OUTER_OPERATOR) + r']')
reg_inner_operator = r'(?:' + r'|'.join(SQ_INNER_OPERATOR) + r').*?(?=' + r'|'.join(SQ_INNER_OPERATOR) + r'|\s*$)'
re_inner_operator = re.compile(reg_inner_operator)

# statement
# re_statement_split = re.compile(r'[;\.]')
re_statement_split = re.compile(r'.*?(?=;|\s\.\s)')
re_statement_inner_keyword = re.compile(r'(?:' + r'|'.join(
    SQ_INNER_KEYWORDS) + r')\s*?[\{\(](?:\(.*\)|[\'\s\w!\"#\$%&()\*+\,-\./\:;<\=>\?@[\]\^_`{|}~])+?[\}\)]')  # (?:\(.*\) # need to check () pairs
re_statement_inner_keyword_filter_special = re.compile(r'(?:FILTER[ ]*[a-zA-Z]{3,}\(.*\))')

re_statement_others = re.compile(r'.*?(?=;|\s\.\s?)')
# re_statement_others = re.compile(r'\b(?:(?<=qpr\:)|(?<=\:))\s?[_a-zA-Z]+\b[ ]+\b.*?\b[ ]')
re_statement_others_last = re.compile(r'(?<=;|\.)[^;\.]*?(?=$)')
re_statement_a = re.compile(r'\?[a-zA-Z]+\s+?\ba\b\s+?(?=[:a-zA-Z])')
# re_statement_a_split = re.compile(r'(?<=[a-zA-Z])\s+?\ba\b\s+?(?=[a-zA-Z])')
re_statement_variable = re.compile(r'(?:^|\s|\b])\?[a-zA-Z]+\b')
re_statement_qpr = re.compile(r'\b(?:(?<=qpr\:)|(?<=\:))\s?[_a-zA-Z]+\b')
re_statement_qpr_constaint = re.compile(r'(?<=\').+(?=\')')
re_statement_content = re.compile(r'(?<=qpr\:).+(?=\s|$)')
# re_statement_content = re.compile(r'qpr\:.+(?=\s|$)')

# re_select_variables = re.compile(r'[\{\(](?:\(.*?\)|[\s\w!\"#\$%&()\*+\,-\./:;<=>\?@[\]\^_`{|}~])+?[\}\)]')
# re_select_variables = re.compile(r'[\{\(](?:\(.+\)|[\s\w!\"#\$%&\(\)\*+\,-\./:;<=>\?@\[\]\^_`\{|\}\~])+?[\}\)]')
re_select_variables = re.compile(r'[\{\(](?:\(.*?\)|[\s\w!\"#\$%&\(\)\*+\,-\./:;<=>\?@\[\]\^_`\{|\}\~])+?[\}\)]')


# function
# re_function_content = re.compile(r'(?:'+r'|'.join(SQ_FUNCTIONS)+r')'+r'.*', re.IGNORECASE)
def re_functions_content(func_name):
    return re.compile(r'(?<=' + func_name + r'\().*?(?=\))', re.IGNORECASE)


re_functions_content = {_: re_functions_content(_) for _ in SQ_FUNCTIONS}
re_function_dependent_variable = re.compile(r'(?<=as)\s+.*(?=\))', re.IGNORECASE)
re_function_distinct = re.compile(r'distinct', re.IGNORECASE)


######################################################################
#   Main Function
######################################################################

def func_sq_common(text, func_name):
    ans = {}
    # print text
    # print func_name
    ans['variable'] = re_functions_content[func_name].search(text).group(0).strip()
    dependent_variable = re_function_dependent_variable.search(text)
    if dependent_variable:
        ans['dependent-variable'] = dependent_variable.group(0).strip()
    ans['type'] = func_name
    return ans


SQP_CURRENT_CONTENT_ID = None


# SQP_CURRENT_CONTENT_FLAG = None

def exception_handler(info):
    # raise Exception(info)
    text = 'SYNTAX IS INCORRECT: ' + info
    if SQP_CURRENT_CONTENT_ID:
        text += '|' + SQP_CURRENT_CONTENT_ID
    print(text)

    return None


class SQParser(object):

    ####################################################
    #   functions for SQ functions
    ####################################################
    def __sqf_func_bind(text):
        # print 'bind'
        # print re_function_content.findall(text)
        pass

    def __sqf_func_bound(text):
        ans = {}
        content = re_functions_content[SQ_FUNCTION_BOUND].search(text)
        if not content:
            exception_handler('Sparql Format Error')
        content = content.group(0).strip()
        ans.setdefault(SQ_FUNCTION_BIND.lower(), content)
        return ans

    def __sqf_func_asc(text):
        pass

    def __sqf_func_desc(text):
        pass

    def __sqf_func_count(text):
        return func_sq_common(text, SQ_FUNCTION_COUNT)

    def __sqf_func_avg(text):
        return func_sq_common(text, SQ_FUNCTION_AVG)

    def __sqf_func_min(text):
        return func_sq_common(text, SQ_FUNCTION_MIN)

    def __sqf_func_max(text):
        return func_sq_common(text, SQ_FUNCTION_MAX)

    def __sqf_func_group_concat(text):
        ans = {}
        values = re_functions_content[SQ_FUNCTION_GROUP_CONCAT].search(text).group(0).strip()
        ans['distinct'] = False
        if re_function_distinct.search(values):
            ans['distinct'] = True
            values = values.replace('distinct').strip()
        values = values.split(';')
        for value in values:
            if '=' not in value:
                ans['variable'] = value.strip()
            else:
                ov = value.split('=')
                ans[ov[0]] = ov[1][1:-1] if '\'' in ov[1] else ov[1]
        dependent_variable = re_function_dependent_variable.search(text)
        if dependent_variable:
            ans['dependent-variable'] = dependent_variable.group(0).strip()
        ans['type'] = 'group-concat'
        return ans

    def __sqf_func_filter_contain(text):
        content = re_brackets_most_s.search(text).group(0).strip()
        source, target = re.split(', ', content)
        source = SQParser.parse_subcomponent(source)
        target = target.strip('"')
        target = target.strip('\'')
        operator = '='
        return source, target, operator

    def __sqf_func_filter_lcase(text):
        return re_brackets_most_s.search(text).group(0).strip()

    SQ_FUNCTIONS_FUNC = {
        SQ_FUNCTION_BIND: __sqf_func_bind,
        SQ_FUNCTION_BOUND: __sqf_func_bound,
        SQ_FUNCTION_ASC: __sqf_func_asc,
        SQ_FUNCTION_DESC: __sqf_func_desc,
        SQ_FUNCTION_COUNT: __sqf_func_count,
        SQ_FUNCTION_AVG: __sqf_func_avg,
        SQ_FUNCTION_MIN: __sqf_func_min,
        SQ_FUNCTION_MAX: __sqf_func_max,
        SQ_FUNCTION_GROUP_CONCAT: __sqf_func_group_concat,
        SQ_FUNCTION_FILTER_CONTAINS: __sqf_func_filter_contain,
        SQ_FUNCTION_FILTER_LCASE: __sqf_func_filter_lcase
    }

    ####################################################
    #   Outer Component Functions
    ####################################################

    def __cp_func_prefix(parent_ans, text):
        pass

    def __cp_func_select(parent_ans, text):
        # SELECT ?cluster ?ad
        # SELECT ?business  (count(?ad) AS ?count)(group_concat(?ad;separator=',') AS ?ads)
        text = ' '.join(text.strip().split(' ', 1)[1:])  # remove keyword
        ans = {}
        ans['variables'] = []
        # print text
        variable_fileds = re_select_variables.findall(text)
        # print "variable_fileds:", variable_fileds
        for variable_filed in variable_fileds:
            text = text.replace(variable_filed, '').strip()
        variable_fileds += text.split(' ')

        # print variable_fileds
        for variable_filed in variable_fileds:
            # variables in function
            # print variable_filed
            is_func = False
            for func_name in SQ_FUNCTIONS:
                # print func_name, variable_filed
                if func_name + '(' in variable_filed.lower():
                    func_rtn = SQParser.SQ_FUNCTIONS_FUNC[func_name](variable_filed)
                    ans['variables'].append(func_rtn)
                    is_func = True
                    break

            if not is_func:
                ans['variables'].append({'variable': variable_filed.strip(), 'type': 'simple'})

        parent_ans.setdefault(SQ_KEYWORD_SELECT.lower(), ans)

    def __cp_func_where(parent_ans, text):
        ans = {}
        # print '__cp_func_where', text
        # print text.encode('utf-8')

        # find all inner keyworkd
        statements = [_.strip() for _ in re_statement_inner_keyword.findall(text)]
        statements += re_statement_inner_keyword_filter_special.findall(text)
        for statement in statements:
            text = text.replace(statement, '')

        # print text
        # print statements

        # print text.encode('utf-8')
        statements += [_.strip() for _ in re_statement_others.findall(text) if _.strip() != '' and _.strip() != '.']

        # print text
        # print statements

        for statement in statements:
            text = text.replace(statement, '')

        if re_statement_others_last.search(text):
            last = re_statement_others_last.search(text).group(0).strip()
            # print 'last', last
            if last != '':
                statements.append(last)

        # print re_statement_others_last.findall(text)[0].strip()

        # text = text.replace('.', '').replace(';', '').strip()
        # statements.append(text)
        # print statements

        # print re_statement_others.findall(text)
        # statements = re_statement_split.split(text)
        # statements = [_.strip() for _ in re_statement_split.findall(text) if _ != '']
        for statement in statements:
            # print 'statement:', statement.encode('ascii', 'ignore')
            SQParser.parse_statement(ans, statement.strip())
        parent_ans.setdefault(SQ_KEYWORD_WHERE.lower(), ans)
        # return ans

    def __cp_func_order(parent_ans, text):
        parent_ans.setdefault(SQ_EXT_GOL, {})
        parent_ans[SQ_EXT_GOL][SQ_EXT_ORDER_VARIABLE] = re_variable.search(text).group(0).strip()

        if re_functions_content[SQ_FUNCTION_ASC].search(text):
            parent_ans[SQ_EXT_GOL][SQ_EXT_SORTED_ORDER] = SQ_FUNCTION_ASC.lower()
        elif re_functions_content[SQ_FUNCTION_DESC].search(text):
            parent_ans[SQ_EXT_GOL][SQ_EXT_SORTED_ORDER] = SQ_FUNCTION_DESC.lower()

    def __cp_func_group(parent_ans, text):
        parent_ans.setdefault(SQ_EXT_GOL, {})
        parent_ans[SQ_EXT_GOL][SQ_EXT_GROUP_VARIABLE] = re_variable.search(text).group(0).strip()

    def __cp_func_limit(parent_ans, text):
        parent_ans.setdefault(SQ_EXT_GOL, {})
        if re_continues_digits.search(text):
            digits = int(re_continues_digits.search(text).group(0).strip())
            if digits > 0:
                parent_ans[SQ_EXT_GOL][SQ_EXT_LIMIT] = digits

    OUTER_COMPONENT_FUNC = {
        SQ_KEYWORD_PREFIX: __cp_func_prefix,
        SQ_KEYWORD_SELECT: __cp_func_select,
        SQ_KEYWORD_WHERE: __cp_func_where,
        SQ_KEYWORD_ORDER: __cp_func_order,
        SQ_KEYWORD_GROUP: __cp_func_group,
        SQ_KEYWORD_LIMIT: __cp_func_limit
    }

    ####################################################
    #   Inner Component Functions
    ####################################################

    def __cp_func_filter(text):
        # print text
        keyword = re_keyword.match(text).group(0).strip()

        if re_statement_inner_keyword_filter_special.search(text):
            source, target, operator = SQParser.parse_subcomponent(text)
            ans = {}
            ans[SQ_EXT_VARIABLE] = source
            ans[SQ_EXT_CONSTAINT] = target
            ans[SQ_EXT_OPERATOR] = operator
            return ans
        else:
            if re_brackets_most_b.search(text):
                text = re_brackets_most_b.search(text).group(0).strip()
            elif re_brackets_most_s.search(text):
                text = re_brackets_most_s.search(text).group(0).strip()

            ans = {}
            for op in SQ_OUTER_OPERATOR:
                if op in text:
                    ans.setdefault(SQ_EXT_OPERATOR, [])
                    ans[SQ_EXT_OPERATOR].append(SQ_OPERATOR_MAPPING[op].lower())

            component = [_.strip() for _ in re_outer_operator_split.split(text) if _ != '']

            subc_rtn = SQParser.parse_subcomponents(component)
            # print subc_rtn
            if len(subc_rtn) > 0:
                ans.setdefault(SQ_EXT_CLAUSES, [])
                ans[SQ_EXT_CLAUSES] += subc_rtn

            # content = re_statement_content.search(text).group(0).strip()
            # print component
            return ans

    def __cp_func_optional(text):
        content = re_statement_content.search(text).group(0).strip()
        if not content:
            exception_handler('Sparql Format Error')
        clause = SQParser.parse_content(content)
        clause[SQ_EXT_OPTIONAL_FLAG] = True
        return clause

    def __cp_func_bind(text):
        pass

    INNER_COMPONENT_FUNC = {
        SQ_KEYWORD_FILTER: __cp_func_filter,
        SQ_KEYWORD_OPTIONAL: __cp_func_optional,
        SQ_KEYWORD_BIND: __cp_func_bind
    }

    ####################################################
    #   Parse Functions
    ####################################################

    @staticmethod
    def parse_content(text):
        ans = {}
        # text = text.split(' ')
        # print 'parse_content: ', text.strip()
        text = text.strip().split(' ', 1)

        # predicate = re_statement_qpr.search(text).group(0).strip()
        # constraint = re_statement_qpr_constaint.search(text).group(0).strip()
        predicate = text[0]
        constraint = text[1]

        constraint = constraint.replace('\'', '')
        # constraint = constraint[1:-1] if '\'' in constraint else constraint
        ans[SQ_EXT_PREDICATE] = predicate
        ans.setdefault(SQ_EXT_OPTIONAL_FLAG, False)
        if '?' in constraint:
            ans[SQ_EXT_VARIABLE] = constraint
        else:
            ans[SQ_EXT_CONSTAINT] = constraint
        return ans

    @staticmethod
    def parse_statement(ans, text):
        # print text
        # print re_statement_a.findall(text)
        if re_statement_a.search(text):
            # print text
            ans[SQ_EXT_TYPE] = re_statement_qpr.search(text).group(0).strip()
            ans[SQ_EXT_VARIABLE] = re_statement_variable.search(text).group(0).strip()
        elif len(re_inner.findall(text)) > 0:
            for component in re_inner.findall(text):
                keyword = re_keyword.match(component).group(0).strip()
                content = component

                # print 'component:', component

                if keyword == SQ_KEYWORD_FILTER:
                    icf_rtn = SQParser.INNER_COMPONENT_FUNC[keyword](content)
                    ans.setdefault(SQ_EXT_FILTERS, [])
                    ans[SQ_EXT_FILTERS].append(icf_rtn)
                else:

                    if re_brackets_most_b.search(component):
                        content = re_brackets_most_b.search(component).group(0).strip()
                    elif re_brackets_most_s.search(component):
                        content = re_brackets_most_s.search(component).group(0).strip()

                    icf_rtn = SQParser.INNER_COMPONENT_FUNC[keyword](content)
                    if keyword == SQ_KEYWORD_OPTIONAL:
                        ans.setdefault(SQ_EXT_CLAUSES, [])
                        ans[SQ_EXT_CLAUSES].append(icf_rtn)

        else:
            # print 'parse_statement:', text, len(re_statement_qpr.findall(text))
            content = re_statement_content.search(text)

            # global SQP_CURRENT_CONTENT_ID
            # print SQP_CURRENT_CONTENT_ID
            # if SQP_CURRENT_CONTENT_ID == '24':
            #     print content

            if not content or len(re_statement_qpr.findall(text)) > 1:
                exception_handler('Sparql Format Error')
            content = content.group(0).strip()
            ans.setdefault(SQ_EXT_CLAUSES, [])
            ans[SQ_EXT_CLAUSES].append(SQParser.parse_content(content))

    @staticmethod
    def parse_inner_operator(op_name, text):
        def clean_item_content(text):
            return text.replace('\'', '').strip()

        ans = {}
        kv = text.strip().split(op_name)

        ans.setdefault(SQ_EXT_VARIABLE, clean_item_content(kv[0]))
        ans.setdefault(SQ_EXT_OPERATOR, op_name)
        ans.setdefault(SQ_EXT_CONSTAINT, clean_item_content(kv[1]))
        return ans

    @staticmethod
    def parse_subcomponent(text):
        # print text
        # functions or condition statement

        # handle functions
        for func_name in SQ_FUNCTIONS:
            # if func_name in text:
            if func_name + '(' in text.lower():  # for contains and lcase, into lower case
                return SQParser.SQ_FUNCTIONS_FUNC[func_name](text)
        # print SQ_INNER_OPERATOR
        # else handle conditions
        for op_name in SQ_INNER_OPERATOR:
            if op_name in text:
                return SQParser.parse_inner_operator(op_name, text)

        exception_handler('Sparql Format Error')

    @staticmethod
    def parse_subcomponents(subcomponents):
        ans = []
        for subcomponent in subcomponents:
            ans.append(SQParser.parse_subcomponent(subcomponent))
        return ans

    @staticmethod
    def parse_components(components):
        ans = {}
        for (key, content) in components.items():
            # print 'key:', key
            # print 'content:', content.encode('ascii', 'ignore')
            # ans[key] = SQParser.OUTER_COMPONENT_FUNC[key](content)
            SQParser.OUTER_COMPONENT_FUNC[key](ans, content)
        return ans

    ####################################################
    # Schema Functions
    ####################################################

    @staticmethod
    def parse_schema_default(**params):
        input_path = params['input_path'] if 'input_path' in params else None
        output_path = params['output_path'] if 'output_path' in params else None
        target_component = params['target_component'] if 'target_component' in params else None
        has_title = params['has_title'] if 'has_title' in params else None

        global SQP_CURRENT_CONTENT_ID
        with open(input_path, 'rb') as file_handler:
            json_obj = json.load(file_handler)
            if has_title:
                for value in json_obj.values():
                    for (k, v) in value.iteritems():
                        SQP_CURRENT_CONTENT_ID = k
                        value[k]['parsed'] = SQParser.parse_string(v['sparql'], target_component=target_component)
            else:
                for (k, v) in json_obj.iteritems():
                    SQP_CURRENT_CONTENT_ID = k
                    k['parsed'] = SQParser.parse_string(v['sparql'], target_component=target_component)

        if output_path:
            file_handler = open(output_path, 'wb')
            file_handler.write(json.dumps(json_obj, sort_keys=True, indent=4))
            file_handler.close()

    @staticmethod
    def parse_schema_payload(**params):

        def parse_payload_json(json_obj):
            ans = []
            for i, query in enumerate(json_obj['SPARQL']):
                parsed = {k: v for (k, v) in json_obj.iteritems() if k != 'SPARQL'}
                parsed['id'] += '-' + str(i + 1)
                parsed['SPARQL'] = SQParser.parse_string(query)
                ans.append(parsed)
            return ans

        input_path = params['input_path']
        output_path = params['output_path']

        ans = []
        with open(input_path, 'r') as file_handler:
            for sparql_query_json_obj in json.load(file_handler):
                ans += parse_payload_json(sparql_query_json_obj)

        if output_path:
            file_handler = open(output_path, 'wb')
            file_handler.write(json.dumps(ans, sort_keys=True, indent=4))
            file_handler.close()

    ####################################################
    # Entry Point
    ####################################################

    @staticmethod
    def parse_string(text, target_component=None):
        components = {re_keyword.match(_).group(0).strip(): re_brackets_most_b.search(_).group(
            0).strip() if re_brackets_most_b.search(_) else _ for _ in re_outer.findall(text)}
        ans = SQParser.parse_components(components)
        if target_component:
            ans = ans[target_component]
        return ans

    @staticmethod
    def parse_json(input_path=None, output_path=None, target_component=None, has_title=False, schema=None):

        if not schema:
            SQParser.parse_schema_default(input_path=input_file, output_path=output_file,
                                          target_component=target_component, has_title=has_title)
        else:
            if schema == SQ_SCHEMA_PAYLOAD:
                SQParser.parse_schema_payload(input_path=input_file, output_path=output_file)

    @staticmethod
    def parse(text, **params):
        pass


if __name__ == '__main__':

    import sys
    import argparse

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('-i', '--input_file', required=False)
    arg_parser.add_argument('-o', '--output_file')
    arg_parser.add_argument('-c', '--target_component', required=False)
    arg_parser.add_argument('-t', '--has_title', required=False)
    arg_parser.add_argument('-s', '--str_input', required=False)
    arg_parser.add_argument('-m', '--schema', required=False)

    args = arg_parser.parse_args()

    input_file = str(args.input_file)
    output_file = str(args.output_file)
    target_component = str(args.target_component) if args.target_component else None
    has_title = args.has_title if args.has_title else False
    str_input = args.str_input
    schema = args.schema

    if str_input:
        print
        json.dumps(SQParser.parse_string(str_input, target_component=target_component), indent=4)
    else:
        SQParser.parse_json(input_path=input_file, output_path=output_file, target_component=target_component,
                            has_title=has_title, schema=schema)




