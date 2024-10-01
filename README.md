# Digital Health Fellowship

This is the code outlining a potential modelling pipeline for **Using digital phenotyping to improve research studies of people at clinical high risk of severe mental disorders**. This is largely focused on the analysis for WP2. While the pipeline for WP3 will largey be similar to this, the training listed as part of my fellowship plans will inform the processing and specific considerations when incorporating longitudinal data.

This script:

* Generates a synthetic dataset that is representative of the processed dataset I will be using in WP2
* Sets up a repeated nested cross validation framework function for logistic regression and random forests
  - Within each inner fold is a semi-supervised learning framework where an initial model is fitted on the labelled data, used to estimate risk on the unlabelled data and these risk estimates that have highest confidence are added to the training data as pseudolabels.
  - The new combined training data is used to fit a new model and the same process is repeated until fewer than 10 confident pseudolabels are generated within any iteration.
  - A final inner model is fitted using the best hyperparameters.
  - This model is then applied to the test data (solely comprised of labelled data) and performance is evaluated.
  - Performance is then assessed in the test data stratifying by ethnicity and gender to explore potential algorithmic bias. 
* An evaluation framework is applied considering discrimination, calibration, algorithmic bias, model complexity, service user feedback and PPI input.
