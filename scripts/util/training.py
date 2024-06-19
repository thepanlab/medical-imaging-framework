def get_list_of_epochs(param_epochs, is_outer, test_subjects):
    """
    Return list of epochs. If unique value is given it repeats
    according to the length of test_subjects list. Otherwise,
    it returns lists of epochs for each subject test

    Args:
        param_epochs (int or list): epochs value(s)
        is_outer (bool): If this is of the outer loop.
        test_subjects (list): list of test subjects
    
    Returns:
        (list) list of epochs
        
    Raises:
        ValueError: if param_epochs is a list when inner loop
        ValueError: if len of epochs != len test subjects
    """
    
    b_single_epoch = True
    
    if isinstance(param_epochs, int):
        epochs = param_epochs
    elif isinstance(param_epochs, list):
        # if len of list is one and inner loop, extract value
        if len(param_epochs) == 1:
            epochs = param_epochs[0]
        # if len of list is greater than one and inner loop, raise ValueError
        elif len(param_epochs) > 1 and is_outer == False:
            raise ValueError("For inner loop, you should have only one value for epoch")

        # Check that the list of epochs is the same length as the list of subjects
        if len(param_epochs) != len(test_subjects):
            raise ValueError(f"Length of list of epochs is :{len(param_epochs)},"+
                                f"length of test_subjects is {len(test_subjects)}")
        else:
            b_single_epoch = False
        
    if b_single_epoch:
        l_epochs = [epochs] * len(test_subjects)
    else:
        l_epochs = param_epochs
            
    return l_epochs