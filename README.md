# HHAWMD

HHAWMD is a hierarchical hypergraph learning method in association-weighted heterogeneous network for miRNA-disease associations identification. HHAWMD first collects information from multiple data sources, including changes in expression level of disease-related miRNAs, miRNA sequence information, disease semantic information, and so on. HHAWMD then uses multi-source information as the attribute features of edges to construct the miRNA-disease heterogeneous graph. Next, HHAWMD extracts the subgraph of each miRNA-disease pair from the heterogeneous graph and builds the hyperedge between each miRNA-disease pair. Finally, HHAWMD applies node-aware attention and hyperedge-aware attention to aggregate various semantic information, assisting hyperedge learning and association identification.

# Requirements
  * Python 3.7 or higher
  * PyTorch 1.8.0 or higher
  * GPU (default)

# Data
  * Download associated data and similarity data.
  * Multiple similarity calculations are detailed in the supplementary material.

# Running  the Code
  * Execute ```python main.py``` to run the code.
  * Parameter state='valid'. Start the 5-fold cross validation training model.
  * Parameter state='test'. Start the independent testing.

# Note
```
 Pay attention to the use of all files.
 Note the version matching of python files.
```
