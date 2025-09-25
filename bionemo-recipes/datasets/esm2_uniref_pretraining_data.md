---
license: cc-by-4.0
tags:
  - bio
pretty_name: ESM-2 Uniref Pretraining Data
task_categories:
  - fill-mask
task_ids:
  - masked-language-modeling
dataset_info:
  features:
    - name: ur50_id
      dtype: string
    - name: ur90_id
      dtype: string
    - name: sequence
      dtype: string
  splits:
    - name: train
      num_examples: 187382018
    - name: valid
      num_examples: 328360
---

# ESM-2 Uniref Pretraining Data

## Dataset Description:

UniRef, or UniProt Reference Clusters, are databases of clustered protein sequences from the UniProt Knowledgebase (UniProtKB) that group similar sequences to reduce redundancy and make data easier to work with for biological research. It offers different levels of clustering (UniRef100, UniRef90, and UniRef50) based on sequence identity, with each cluster containing a representative sequence, a count of member proteins, and links to detailed functional annotations in the UniProtKB.

We are releasing a subset of UniRef (UniRef50 + UniRef90) that was used for pretraining ESM-2nv models, with the following modifications. We removed the artificial sequences from UniRef50 and UniRef90 and created our own training and validation sets. We further performed MMseqs (Many-against-Many sequence searching) clustering on these datasets.

This dataset is ready for commercial/non-commercial use.

## Dataset Owner(s):

The UniRef dataset is owned and maintained by the UniProt Consortium, a collaboration between three major bioinformatics institutes: European Bioinformatics Institute (EBI), SIB Swiss Institute of Bioinformatics, and Protein Information Resource (PIR).

## Dataset Creation Date:

July 19, 2024.

## License/Terms of Use:

Governing Terms: This dataset is licensed under the [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/legalcode.en) (CC BY 4.0).

## Intended Usage:

ESM2-nv is using this dataset for model pretraining. This dataset can be used by protein designers, structural biologists, bioengineers, computational biologists and protein engineers for pretraining other similar models.

## Dataset Characterization

**Data Collection Method**

- Human

**Labeling Method**

- N/A

## Dataset Format

The dataset is provided in the standard FASTA format, with one entry for each representative protein sequence from the UniRef90 and UniRef50 clusters. Each entry consists of a header line and the protein sequence itself.

## Dataset Quantification

187,382,018 training sequences, chosen from UniRef90 representative sequences. 328,360 validation sequences, chosen from UniRef50 representative sequences.

The total data storage is approximately 35GB.

## Reference(s):

1. [Uniprot Reference Clusters (UniRef)](https://www.uniprot.org/uniref)
2. [ESM-2nv 650M](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/esm2nv650m)
3. [ESM-2nv 3B](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/clara/models/esm2nv3b)
4. Original ESM-2 Paper: Lin, Z., Akin, H., Rao, R., Hie, B., Zhu, Z., Lu, W., Smetanin, N., Verkuil, R., Kabeli, O., Shmueli, Y., dos Santos Costa, A., Fazel-Zarandi, M., Sercu, T., Candido, S., & Rives, A. (2023). Evolutionary-scale prediction of atomic level protein structure with a language model. *Science*, *379*(6637), eade2574. [https://doi.org/10.1126/science.ade2574](https://doi.org/10.1126/science.ade2574)

## Ethical Considerations:

NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

Please report security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/).
