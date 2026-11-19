# Leak audit: clean protocol (v3)

The v2 stand-in was trained on the train split plus `train_pool.jsonl`. Groups that touch
the held-out test items were found in the training data. The table gives the counts before
and after removal. Earlier test numbers from v2 are contaminated and are NOT used as
independent results.

- held_out_items: 243
- held_out_groups: 241
- train_split_items: 892
- train_split_items_removed_for_leakage: 37
- pool_rows: 2448
- pool_rows_removed_for_leakage: 184
- pool_exact_matches_with_held_out: 51
- dev_rows_kept: 3119
- dev_groups: 2505
- dev_groups_with_multiple_rows: 371
- held_family_counts: {'intent': 31, 'sentiment': 32, 'nli': 41, 'factoid': 40, 'multihop': 63, 'math': 36}
- dev_family_counts: {'intent': 1705, 'sentiment': 336, 'nli': 587, 'factoid': 160, 'multihop': 187, 'math': 144}
