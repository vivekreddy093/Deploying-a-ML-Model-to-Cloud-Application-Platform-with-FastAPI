# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
- It is a Random Forest classifier developed using scikit-learn.
- It is implemented as part of the Udacity Machine Learning DevOps Nanodegree.
- It is based on the started code provided as part of Nanodegree.

## Intended Use
The model is trained on data from 1994, making it suitable for datasets with a similar distribution &#8212; if your data has a similar distribution, you can use this model to predict whether an income is above or below 50K.

## Training Data
- Source: [Census Income Dataset](https://archive.ics.uci.edu/dataset/20/census+income)
- 80% of it was used for training the model.

## Evaluation Data
20% of aforementioned dataset was used for evaluating the model's performance.

## Metrics
_Please include the metrics used and your model's performance on those metrics._
- Precision: 0.76
- Recall: 0.63
- F1: 0.69

## Ethical Considerations
- Before making this model live, it is very crucial to evaluate potential data/model biases &#8212; we conducted a slice evaluation based on features such as workclass, education, and native country.
- It is important to scrutinize above proxy metrics to understand and mitigate biases, ensuring fair and ethical use of the model.

## Caveats and Recommendations
This model is nothing but a first run and can be improved with additional effort ie., the preprocessing was based on simple exploration, and we can seek for advanced data preprocessing, and model development. For real-world applications, it is recommended to use updated and current datasets to ensure the model's relevance and accuracy.

