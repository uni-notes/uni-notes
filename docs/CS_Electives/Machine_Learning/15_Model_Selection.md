# Selection

Note: Make sure to correct for multiple hypothesis testing

## Model Selection

1. Fit multiple models $g_i$ on the training data and eyeball dev data
2. Use dev data for hyper parameter tuning of each model $g_i$
3. Use external validation data for model selection and obtain $g^*$
4. Combine the training and validation data. Refit $g^*$ on this set to obtain $g^{**}$
5. Assess the performance of $g^{**}$ on the test data

Finally, train $g^{**}$ on the entire data to obtain $\hat f$

Check the results on the self-hosted competition