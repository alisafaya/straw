
# Fiction filter

We filter Books3 and PG-Gutenberg using an BOW/XGBoost classifer into Fiction/Non-fiction. We train the XGBoost classifier on the metadata of _BookCorpus v1_.

## Fiction/Non-fiction prediction results

Out of ~280K books in the Books3 corpus of Pile, the classifier labeled ~117K as fiction books. 

The precision of the classifier in detecting fiction books is around 98% on the test set. Detailed performance report of our classifier on the test set can be seen below:

```
Fiction vs Non-fiction:

               precision    recall  f1-score   support

           0      0.954     0.941     0.947       824
           1      0.980     0.985     0.982      2443

    accuracy                          0.974      3267
   macro avg      0.967     0.963     0.965      3267
weighted avg      0.974     0.974     0.974      3267
```
