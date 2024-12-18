# scTrace+

[![PyPI version](https://badge.fury.io/py/scTrace.svg?icon=si%3Apython)](https://pypi.org/project/scTrace/)

## Introduction
`scTrace+` is a computational method to enhance the cell fate inference by integrating the lineage-tracing and multi-faceted transcriptomic similarity information.

<div align=center>
<img alt="scTrace+ workflow" src="tutorial/scTrace.png" width="70%"/>
</div>

## System Requirements
- Python version: >= 3.7

## Installation

The Release version of `scTrace+` python package can be installed directly via pip:
```
pip install scTrace
```

Besides, we provided the develop version of scTrace+. After installing `scStateDynamics` and `node2vec`,
you can run our [tutorial](https://github.com/czythu/scTrace/tree/main/tutorial) 
to perform LT-scSeq data enhancement and cell fate inference steps.
```
pip install scStateDynamics
pip install node2vec
git clone https://github.com/czythu/scTrace.git
```

## Quick Start of LT-scSeq data enhancement

Refer to folder: [tutorial](https://github.com/czythu/scTrace/tree/main/tutorial) for full pipeline.

Example data1: [Larry-Invitro-differentiation](https://cloud.tsinghua.edu.cn/f/1b94b3229f4a4c52985e/?dl=1)

Example data2: [TraCe-seq-tumor](https://cloud.tsinghua.edu.cn/f/dae5b3ff8bd04177bd5f/?dl=1)

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

9. `runFateDE`: Perform differential expression analysis between selected dynamic sub-clusters.

10. `dynamicDiffAnalysis`: Perform differential expression analysis between all dynamic sub-clusters (1 v.s. rest).

## Citation
scTrace+: enhance the cell fate inference by integrating the lineage-tracing and multi-faceted transcriptomic similarity information, Wenbo Guo, Zeyu Chen, Xinqi Li, Jingmin Huang, Qifan Hu, Jin Gu, https://doi.org/10.1101/2024.11.12.623316
