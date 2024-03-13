# BlindNet (CVPR 2024) : Official Project Webpage
This repository provides the official PyTorch implementation of the following paper:
> [**Style Blind Domain Generalized Semantic Segmentation via Covariance Alignment and Semantic Consistence Contrastive Learning**](https://arxiv.org/pdf/2403.06122.pdf])<br>
> Woo-Jin Ahn, Geun-Yeong Yang, Hyun-Duck Choi, Myo-Taeg Lim<br>
> Korea University, Chonnam National University

> **Abstract:**
> *Deep learning models for semantic segmentation often
experience performance degradation when deployed to unseen target domains unidentified during the training phase.
This is mainly due to variations in image texture (i.e. style)
from different data sources. To tackle this challenge, existing domain generalized semantic segmentation (DGSS)
methods attempt to remove style variations from the feature. However, these approaches struggle with the entanglement of style and content, which may lead to the unintentional removal of crucial content information, causing
performance degradation. This study addresses this limitation by proposing BlindNet, a novel DGSS approach that
blinds the style without external modules or datasets. The
main idea behind our proposed approach is to alleviate the
effect of style in the encoder whilst facilitating robust segmentation in the decoder. To achieve this, BlindNet comprises two key components: covariance alignment and semantic consistency contrastive learning. Specifically, the
covariance alignment trains the encoder to uniformly recognize various styles and preserve the content information
of the feature, rather than removing the style-sensitive factor. Meanwhile, semantic consistency contrastive learning
enables the decoder to construct discriminative class embedding space and disentangles features that are vulnerable to misclassification. Through extensive experiments,
our approach outperforms existing DGSS methods, exhibiting robustness and superior performance for semantic segmentation on unseen target domains.* <br>
