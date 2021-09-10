from tables import TableVisitor
import re
import sqlparser
from sqlparser import ENodeType
import pandas as pd

# counts subqueries
sub_count = 0

# flag for nested aggregation
nested_aggregation = False

# count function calls
num_of_function_calls = 0

# count table names
number_of_unique_table_names = 0

# count selected columns
number_of_selected_columns = 0

# count predicates
number_of_predicates = 0

# count predicate table names
number_of_predicate_table_names = 0

# subqueries
subqueries = 0


# Translates a node_type ID into a string
def translate_node_type(node_type):
    for k, v in ENodeType.__dict__.items():
        if v == node_type:
            return k
    return None


# Print node and traverse recursively
def process_node(node, depth=0, name='root'):
    # original function used to print the syntax tree of an SQL query made by https://github.com/timogasda
    # modified by Antonis Papadakis so that nested queries could be counted and the nestedness level could be identified
    # print attributes like ints and strings

    if not isinstance(node, sqlparser.Node):
        # print("%s%s: '%s'" % ("   " * depth, name, str(node)))
        return

        # print node attribute
    if node is not None and isinstance(node, sqlparser.Node):

        node_text = node.get_text()
        # print(depth, " %s%s: %s (%d), text: '%s'" % ("   " * depth, name, translate_node_type(node.node_type), node.node_type, node_text))

        global sub_count
        if "subQueryNode" in name and 'None' not in translate_node_type(node.node_type):
            sub_count += 1

        if "functionCall" in name and 'None' not in translate_node_type(node.node_type):
            if sub_count >= 1:
                global nested_aggregation
                nested_aggregation = True
            global num_of_function_calls
            num_of_function_calls += 1

    # Go through the list (if the current node is a list)
    if node.node_type == ENodeType.list:
        for i, subnode in enumerate(node.list):
            process_node(subnode, depth + 1, 'list_item#%s' % i)
        return

    # Iterate through all attributes
    for k, v in node.__dict__.items():
        process_node(v, depth + 1, k)


class Analiser(pd.DataFrame):

    def __init__(self, data):
        # replace newlines with spaces
        data.replace(to_replace=[r"\\t|\\n|\\r", "\t|\n|\r"], value=["", ""], regex=True, inplace=True)

        # -------------------------------------------------------
        # tokens = parser.tokenize(query)

    def analisis(self, data):

        rows, cols = data.shape

        for r in range(0, rows):
            print(r)
            # q = "SELECT * FROM students WHERE GPA > (SELECT AVG(GPA) FROM students WHERE AGE < (SELECT MAX(AGE) FROM students) );"
            q = data.statement[r]
            print(q)
            self.analise_all(q)

    def analise_all(self, q, parent=True):
        """
        this function performs the query analisis i.e. number of words, joins, functions etc.
        :param q: this is the query to be analised
        :param parent: this is the parameter that checks if we are checking the parent query or a left or right node in
        it

        """
        global num_of_function_calls, sub_count

        if q is not None:
            parser = sqlparser.Parser(vendor=0)
            ch = parser.check_syntax(q)
        else:
            return

        if ch[0] == 0:

            # get the number of characters in a query statement
            if parent:
                num_of_chars = len(q)

            # count the number of words in a query
            if parent:
                temp = q
                word_list = temp.split()
                num_of_words = len(word_list)

            # count joins
            if parent:
                num_of_joins = q.count('JOIN')
                num_of_joins += q.count('join')
                num_of_joins += q.count('Join')

            # if the parser cant extract any statement from the query return
            if parser.get_statement_count() == 0:
                return

            # Get first the statement from the query
            stmt = parser.get_statement(0)
            if stmt is not None:
                # Get root node
                root = stmt.get_root()

                # Create new visitor instance from root
                visitor = TableVisitor(root)
                # Traverse the syntax tree
                # visitor.traverse()

                for table in visitor.used_tables:
                    if table['alias']:
                        print("%s => %s @ %s" % (table['alias'], table['name'], table['position']))
                    else:
                        print("%s @ %s" % (table['name'], table['position']))

                it = root.__dict__  # getting the sqlnode item dictionary

                # count number of unique table names
                # print(visitor.used_tables)
                # number_of_unique_table_names = len(visitor.used_tables)
                # print(it)

                # important condition to check the existence of a UNION operator
                if 'leftNode' and 'rightNode' in it and it['leftNode'] and it['rightNode'] is not None:
                    # if there is a right and left node in the query then recursively process each part of the query

                    q1 = it['leftNode'].get_text()
                    self.analise_all(q1, parent=False)

                    q2 = it['rightNode'].get_text()
                    self.analise_all(q2, parent=False)

                # check if there are any tables mentioned in the query and if so proceed
                global number_of_unique_table_names
                tables_used = []

                if 'fromTableList' in it and it['fromTableList'] is not None:
                    tables_used = it['fromTableList'].get_text().strip().split(', ')

                    number_of_unique_table_names += len(tables_used)

                    # check if there are any database function calls used
                    for j in tables_used:
                        if "(" in j:
                            num_of_function_calls += 1

                global number_of_predicates
                global number_of_predicate_table_names
                global number_of_selected_columns

                # check for the result columns in the query
                if 'resultColumnList' in it:
                    # number of selected columns
                    if it['resultColumnList'] is not None:
                        number_of_selected_columns += len(it['resultColumnList'].get_text().strip().split(','))

                    # number of predicates
                    if it['whereCondition'] is not None:
                        re_str_pr = re.split(',|AND|OR|NOT', it['whereCondition'].get_text()[6::])

                        number_of_predicates += len(re_str_pr)

                        # number of predicate table names
                        # we use regular expression splits to get the predicates
                        re_str = re.split('[<>]|=|[<>]=|==|AND|OR|NOT', it['whereCondition'].get_text()[6::])

                        keep_track = []
                        for predicate_part in re_str:
                            if '.' in predicate_part:
                                t = predicate_part.strip().split('.')
                                table_name = t[0]
                                for j in tables_used:
                                    if table_name in j:  # if predicate table name matches the one in the query table matrix and we have not already searched for that table name
                                        if table_name not in keep_track:
                                            number_of_predicate_table_names += 1
                                            keep_track.append(table_name)
                                            break
                                        else:
                                            break

                            elif predicate_part in tables_used:
                                number_of_predicate_table_names += 1

                    else:
                        number_of_predicates = 0
                        number_of_predicate_table_names = 0
                else:
                    number_of_predicates = 0
                    number_of_predicate_table_names = 0

                # nestedness level
                # number of subqueries
                process_node(root)
                global subqueries
                subqueries += sub_count

                # subquery aggregation
                if parent:
                    global nested_aggregation
                    print("num of chars: ", num_of_chars, "\nnum of words: ", num_of_words,
                          "\nnum of function calls",
                          num_of_function_calls, "\nnum of joins: ", num_of_joins, "\nnum of unique table names: ",
                          number_of_unique_table_names,
                          "\nnum of selected columns: ", number_of_selected_columns,
                          "\nnum of predicates: ", number_of_predicates, "\nnumber of predicate table names: ",
                          number_of_predicate_table_names,
                          "\nnumber of subqueries: ", subqueries, "\nnested aggregation: ", nested_aggregation)

                    sub_count = 0
                    subqueries = 0
                    num_of_function_calls = 0
                    number_of_unique_table_names = 0
                    number_of_predicates = 0
                    number_of_predicate_table_names = 0
                    number_of_selected_columns = 0
                    nested_aggregation = False
            else:
                print("There is no query in this entry\n")
