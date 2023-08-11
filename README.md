## Abstract

Proteins play a crucial role in biological processes, and accurately comparing their structures is essential for advancing large-scale biological research. Existing protein comparison methods struggle to efficiently integrate the wealth of information contained in protein sequences and structures. In this paper, we propose a novel framework for embedding protein graphs in geometric vector spaces, leveraging Graph Neural Networks (GNNs) and Large Language Models (LLMs). Our framework generates structure- and sequence-aware protein representations, enabling efficient and accurate protein structure comparison. We demonstrate the effectiveness of our embeddings in various tasks, including protein structure classification, with significant improvements compared to existing methods. Our approach has applications in drug prioritization, drug re-purposing, disease sub-type analysis, and more.

## Introduction

Proteins are organic macro-molecules composed of twenty standard amino acids and are involved in numerous vital biological functions. The comparison of protein structures is essential for tasks such as protein structure prediction, protein-protein docking, and structure-based function prediction. However, existing protein comparison methods face challenges in efficiently integrating the information contained in protein sequences and structures.

Current protein comparison methods can be categorized as alignment-based or alignment-free. Alignment-based methods aim to find the optimal superposition of protein structures, followed by measuring the distance between the aligned residues. Alignment-free methods represent proteins using descriptors and measure the distance between pairs of descriptors. However, these methods often struggle to handle large-scale protein databases and fail to incorporate both sequence and structure information effectively.

To address these limitations, we propose a novel framework for protein graph embedding in geometric vector spaces. Our approach leverages the power of Graph Neural Networks (GNNs) and Large Language Models (LLMs) to generate structure- and sequence-aware protein representations. By embedding protein graphs, we enable efficient and accurate protein structure comparison, overcoming the computational challenges of existing methods. Additionally, we directly incorporate TM-scores, a widely used metric for structural similarity, into our loss function formulation.

## Key Contributions

    We introduce a novel framework that combines Graph Neural Networks (GNNs) and Large Language Models (LLMs) for protein graph embedding in geometric vector spaces.
    Our framework efficiently computes similarities between protein structures, enabling quick and accurate protein comparison without the need for computationally expensive superposition calculations.
    We evaluate the effectiveness of our embeddings in various tasks, including protein structure classification, and demonstrate significant improvements compared to existing state-of-the-art methods.
    Our approach finds applications in drug prioritization, drug re-purposing, disease sub-type analysis, and other areas of bioinformatics and drug discovery.

## Getting Started

To replicate our results and utilize our protein graph embedding framework, follow these steps:

    Install the required dependencies listed in requirements.txt.
    Download the protein data in PDB format and preprocess it to obtain the graph representations.
    Create a CSV file containing protein pair information and TM-scores for training and evaluation.
    Adjust the hyperparameters, such as the distance function and output dimensions, according to your specific requirements.
    Run the provided scripts to train the model and evaluate its performance.
    Utilize the trained model for various downstream tasks, such as protein structure classification.

## Results and Discussion

We conducted extensive experiments to evaluate the effectiveness of our protein graph embedding framework. Our embeddings achieved remarkable results in various tasks, including protein structure classification, outperforming state-of-the-art methods. Specifically, compared to existing work, our approach showed an average F1-Score improvement of 26% on out-of-distribution (OOD) samples and 32% on samples from the same distribution as the training data. These results highlight the power of our embeddings in capturing structural similarities and their potential impact on areas such as drug prioritization, drug re-purposing, and disease sub-type analysis.

## Conclusion

In this paper, we presented a novel framework for protein graph embedding in geometric vector spaces. Our approach leverages Graph Neural Networks (GNNs) and Large Language Models (LLMs) to generate structure- and sequence-aware protein representations. By embedding protein graphs, we enable efficient and accurate protein structure comparison, overcoming the limitations of existing methods. The effectiveness of our embeddings was demonstrated through experiments, showcasing their superior performance in protein structure classification. We believe our approach has significant potential in advancing large-scale biological research and finding applications in various domains, including bioinformatics and drug discovery.
