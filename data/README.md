This folder contains three processed datasets used in paper "Mining Entity Synonyms with Efficient Neural Set Generation". 

Each dataset contains five files:

* combined.embed: the word2vec embedding for each term in the vocabulary
* combined.fastText-no-subword.embed: the fastText embedding (with subword information disabled) for each term in the vocabulary
* combined.fastText-with-subword.embed: the fastText embedding (with subword information enabled) for each term in the vocabulary
* train-cold.set: training synonym sets, each line represents a synonym set and is of format "<synset-id> {elements in the synset}". 
* test.set: testing synonym sets, each line represents a synonym set and is of format "<synset-id> {elements in the synset}". 

Note 1: The above five files are all you need to reproduce the paper results. If you want to extract more term features (beyond the pre-trained term embeddings), you can download the raw text corpus from: [http://bit.ly/SynSetMine-dataset](http://bit.ly/SynSetMine-dataset).

Note 2: Although each term here is represented as "<entity-surface-name>||<freebase-id>", the freebase-id information is not used to train the term embedding or to learn the model. 



