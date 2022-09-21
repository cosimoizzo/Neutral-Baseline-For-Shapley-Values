# Neutral-Baseline-For-Shapley-Values

This package provides replication codes for the following paper.

Title: "A Baseline for Shapley Values in MLPs: from Missingness to Neutrality"

Abstract: "Deep neural networks have gained momentum based on their accuracy, but their interpretability is often criticised. As a result, they are labelled as black boxes. In response, several methods have been proposed in the literature to explain their predictions. Among the explanatory methods, Shapley values is a feature attribution method favoured for its robust theoretical foundation. However, the analysis of feature attributions using Shapley values requires choosing a baseline that represents the concept of missingness. An arbitrary choice of baseline could negatively impact the explanatory power of the method and possibly lead to incorrect interpretations. In this paper, we present a method for choosing a baseline according to a neutrality value: as a parameter selected by decision-makers, the point at which their choices are determined by the model predictions being either above or below it. Hence, the proposed baseline is set based on a parameter that depends on the actual use of the model. This procedure stands in contrast to how other baselines are set, i.e. without accounting for how the model is used. We empirically validate our choice of baseline in the context of binary classification tasks, using two datasets: a synthetic dataset and a dataset derived from the financial domain."

Accepted as conference paper at "ESANN 2021 proceedings, European Symposium on Artificial Neural Networks, Computational Intelligence 
and Machine Learning. Online event, 6-8 October 2021, i6doc.com publ., ISBN 978287587082-7." 
https://www.esann.org/sites/default/files/proceedings/2021/ES2021-18.pdf
