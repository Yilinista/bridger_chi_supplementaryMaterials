strategy 1:
softmax_score threshold: 0.9
term frequency is by occurrence (can have multiple occurrences per paper)

strategy 2:
softmax_score threshold: 0.9
term frequency is by paper (max one occurrence per paper)

strategy 3:
softmax_score threshold: 0.9
term frequency is by paper (max one occurrence per paper)
filter out terms that only appear in one paper

strategy 4:
softmax_score threshold: 0.9
term frequency is by paper (max one occurrence per paper)
map terms that end in list of generic words to the term with the generic word removed (if exists)
e.g.: ai task == ai technology == ai system == ai model = ai
generic_words = ['component', 'block', 'module', 'task', 'methodology', 'component', 'technology', 'mechanism', 'approach', 'method', 'technique', 'framework', 'system', 'model', 'algorithm', 'procedure']