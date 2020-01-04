
# Cross-validation

This workflow is a Cross-Validation method built on top of the Train part of `Log(ML)` main workflow.


The YAML configuration is quite simple, you need to enable cross-validation and then specify the cross-validation type and the parameters:
The cross-validation workflow is implemented using SciKit learn's cross validation, on the methods and parameters see [SciKit's documentation](https://scikit-learn.org/stable/modules/cross_validation.html)
```
cross_validation:
    enable: True	# Set this to 'True' to enable cross validation
    # Select one of the following algorithms and set the parameters
    KFold:
        n_splits: 5
    # RepeatedKFold:
    #     n_splits: 5
    #     n_repeats: 2
    # LeaveOneOut:
    # LeavePOut:
    #     p: 2
    # ShuffleSplit:
    #     n_splits: 5
    #     test_size: 0.25
```
