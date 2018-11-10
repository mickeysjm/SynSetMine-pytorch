This folder contains three processed datasets used in paper "Mining Entity Synonyms with Efficient Neural Set Generation". 

Each dataset contains five files:
1. combined.embed: the word2vec embedding for each term in the vocabulary
2. combined.fastText-no-subword.embed: the fastText embedding (with subword information disabled) for each term in the vocabulary
3. combined.fastText-with-subword.embed: the fastText embedding (with subword information enabled) for each term in the vocabulary
4. train-cold.set: training synonym sets, each line represents a synonym set and is of format "<synset-id> {elements in the synset}". 
5. test.set: testing synonym sets, each line represents a synonym set and is of format "<synset-id> {elements in the synset}". 

Note: Although each term here is represented as "<entity-surface-name>||<freebase-id>", the freebase-id information is not used to train the term embedding or to learn the model. 

For the original full datasets that include the raw text corpus, you can download them from: http://bit.ly/SynSetMine-dataset