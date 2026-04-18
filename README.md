# FusionDP: Foundation Model-Assisted Differentially Private Learning for Partially Sensitive Attributes

### Official repository for FusionDP

Datasets in sensitive domains often contain attributes with heterogeneous privacy requirements, where different features are subject to different access controls and compliance policies. For instance, in health datasets, specific identifiers like age and race are protected health information, while clinical measurements and lab results may be freely used for analytics. However, existing differential privacy (DP) mechanisms apply uniform protection across all features, leading to excessive noise injection that degrades the utility of machine learning models.
We propose FusionDP, a framework that enables feature-level privacy control when training over partially sensitive data. FusionDP first leverages foundation models to impute sensitive features from non-sensitive ones, creating a privacy-preserving view of the data. It then uses a modified DP-SGD algorithm that trains on both original and imputed features while formally guaranteeing privacy for sensitive attributes.
We evaluate FusionDP on 4 different classification tasks. Compared to standard DP-SGD baselines, FusionDP significantly improves model accuracy while maintaining rigorous feature-level privacy, demonstrating how exploiting feature-level heterogeneity enhances the privacy-utility tradeoff in sensitive data analytics.

## Problem setting and training pipeline

We consider a scenario where only a subset of features requires privacy protection, while the rest can be used without restrictions.

Formally, let $\mathcal{X}$ denote the data space, where each data point $x \in \mathcal{X}$ can be decomposed into sensitive (private) components $x_{\text{priv}}$ and non-sensitive (public) components $x_{\text{pub}}$, such that $x = (x_{\text{priv}}, x_{\text{pub}})$. 
In real-world applications, $x_{\text{priv}}$ may include demographic attributes, rare diagnoses, education history, or identifiable spans in text that pose greater re-identification risks, or sensitive features specified by users. Meanwhile, $x_{\text{pub}}$ encompasses features like lab results, transactions, or non-identifying text tokens that do not require the same level of protection.

<img width="741" height="733" alt="fusiondp" src="https://github.com/user-attachments/assets/66ebc6e3-4e6a-430d-87bf-7c9cbfecc374" />

FusionDP is a two-step framework to achieve feature-DP with improved utility. The figure illustrates this training pipeline. We first use foundation models to generate hybrid samples where sensitive features are replaced by imputed values. We then train the model with a combined loss objective of public (in green) and private (in red) components.
We clip and add noise only to the gradient of the private loss, which isolates and bounds the contribution of private features.
Under this framework, we improve the private gradient component by leveraging the gradient calibration and proposing a representation-consistency regularizer to align hidden states of original and hybrid inputs. 

## Requirement

Install the environment:

<pre>conda env create -f environment.yml
conda activate fusiondp
</pre>

## Running FusionDP

Take the bank marketing dataset as an example, FusionDP can be ran using the following argument:

run python impute_bank.py to impute sensitive attributes with TabPFN and get bank_train.csv, bank_train_imputed.csv, bank_val.csv, bank_test.csv for training.

<pre> python train_fusiondp_bank.py \
      --one_hot --mode fusiondp --epsilon $eps --epochs 10 --max_grad_norm $c \
          --alpha $a --beta $b </pre>

## Datasets

- [PhysioNet Sepsis Prediction](https://physionet.org/content/challenge-2019/1.0.0/)
- [Bank Marketing](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- [Adult Income](https://archive.ics.uci.edu/dataset/2/adult)
- [Mimic3 Clinical Notes Classification](https://physionet.org/content/mimiciii/1.4/)

## Results

<img width="526" height="354" alt="Screenshot 2026-01-31 at 5 55 54 PM" src="https://github.com/user-attachments/assets/592a1159-4cdb-4819-b365-b96ac668ef0e" />

<img width="1093" height="366" alt="Screenshot 2026-01-31 at 5 56 07 PM" src="https://github.com/user-attachments/assets/c0b0421e-73c8-40e9-80e3-e50f54de6bf4" />
