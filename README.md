# scTrace+

## Introduction
`scTrace+` is a computational method to enhance single-cell lineage tracing data through the kernelized bayesian network.

## System Requirements
- Python version: >= 3.7

## Installation

Currently, we provided the develop version of scTrace+. After installing `scStateDynamics` and `node2vec`,
you can run our demo to perform LT-scSeq data enhancement and cell fate inference steps.
```
pip install scStateDynamics
pip install node2vec
```

The Release version and tutorial of `scTrace+` python package will be updated soon. It can be installed directly via pip:
```
pip install scTrace
```

## Quick Start of LT-scSeq data enhancement

Refer to tutorial/Larry-InvitroDiff.ipynb for full pipeline.

Below are the introduction to important functions, consisting of the main steps in `scTrace+`.

1. `prepareCrosstimeGraph`: Process input time-series dataset, output lineage adjacency matrices
and transcriptome similarity matrices, both within and across timepoints.

2. `prepareSideInformation`: Derive low-dimensional side information matrix with `node2vec` and `rbf kernel`.

3. `trainMF`: Train scLTMF model to predict the missing entries in the original across-timepoint transition matrix.

4. `predictMissingEntries`: Load pretrained scLTMF model and calculate performance evaluation indicators.

5. `prepareScdobj`: Prepare `scStateDynamics` objects and perform clustering method.

6. `visualizeLineageInfo` & `visualizeEnhancedLineageInfo`: Visualize cluster alignment results with Sankey plot. 

7. `assignLineageInfo`: Assign fate information at single-cell level and output a `cell2cluster` matrix according to lineage information.

8. `enhanceFate`: Enhance cell fate information based on hypothesis testing method for single-cell level fate inference.

9. `runFateDE`: Perform differential expression analysis between dynamic sub-clusters.
