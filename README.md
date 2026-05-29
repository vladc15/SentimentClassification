# SentimentClassification

We study five-class sentiment classification for En-
glish and German product reviews. The task is
ordinal: predicting a neighboring star rating is
preferable to predicting a distant one. We there-
fore evaluate models with the competition score
1 − MAE/4 and compare simple embedding base-
lines, several pretrained transformer checkpoints,
title/body variants, and probability-level ensem-
bles. Our final system combines heterogeneous
transformer predictions with weighted averaging
and learned multilayer perceptron (MLP) stacking.
Beyond leaderboard optimization, we analyze the
data distribution, inspect model errors, and explain
representative RoBERTa predictions with LIME
and Integrated Gradients. The best submission
obtains a competition score of 0.910.