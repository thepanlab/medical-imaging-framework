from random import shuffle


def generate_folds(test_subject_list, validation_subject_list, test_subject, do_shuffle):
    """ Generates folds for the subject.
        
        -- Input Parameters ------------------------
        test_subject_list (list of str): A list of test subject names.
        validation_subject_list (list of str): A list of validation subject names.
        test_subject (str): The current test subject name.
        do_shuffle (bool): If the fold list should be shuffled or not.
        --------------------------------------------
        
        -- Returns ---------------------------------
        (list of dict): A list of folds, containing the subjects for testing, validation, and training.
        (int): The number of rotations for the training loop.
        --------------------------------------------
    """
    if validation_subject_list is None:
        return _outer_loop_generation(test_subject_list, test_subject, do_shuffle)
    
    else:
       return _inner_loop_generation(test_subject_list, validation_subject_list, test_subject, do_shuffle)



def _outer_loop_generation(test_subject_list, test_subject, do_shuffle):
    """ Generates folds for the outer loop subject.
        
        -- Input Parameters ------------------------
        test_subject_list (list of str): A list of test subject names.
        test_subject (str): The current test subject name.
        do_shuffle (bool): If the fold list should be shuffled or not.
        --------------------------------------------
        
        -- Returns ---------------------------------
        (list of dict): A list of folds, containing the subjects for testing, validation, and training.
        (int): The number of rotations for the training loop.
        --------------------------------------------
    """
    training_folds = []
    for i, subject in enumerate(test_subject_list):
        
        # For the given test subject, get the training and testing subjects
        if subject == test_subject:
            training_folds.append({'training': [], 'testing': []})
            training_folds[-1]['testing'].append(subject)
            
            # Generate the testing dataset from the other subjects
            for j, item_train in enumerate(test_subject_list):
                if i != j:
                    training_folds[-1]['training'].append(item_train)
    
    # Shuffle the data and get the number of training rotations
    if do_shuffle:
        shuffle(training_folds)
    return training_folds, len(training_folds)



def _inner_loop_generation(test_subject_list, validation_subject_list, test_subject, do_shuffle):
    """ Generates folds for the inner loop subject.
        
        -- Input Parameters ------------------------
        test_subject_list (list of str): A list of test subject names.
        validation_subject_list (list of str): A list of validation subject names.
        test_subject (str): The current test subject name.
        do_shuffle (bool): If the fold list should be shuffled or not.
        --------------------------------------------
        
        -- Returns ---------------------------------
        (list of dict): A list of folds, containing the subjects for testing, validation, and training.
        (int): The number of rotations for the training loop.
        --------------------------------------------
    """
    folds = []
    
    # For this test subject, find all combinations for the testing data
    i = test_subject_list.index(test_subject)
    for j, validation_subject in enumerate(validation_subject_list):
        if i != j:
            subject_fold = {'training': _fill_training_fold(validation_subject_list, i, j), 'validation': [validation_subject], 'testing': [test_subject]}
            folds.append(subject_fold)
    
    # Shuffle the data and get the number of training rotations
    if do_shuffle:
        shuffle(folds)
    return folds, len(folds)



def _fill_training_fold(subject_list, i, j):
    """ Fills the training fold for some subject.

        -- Input Parameters ------------------------
        subject_list (list of str): A list of possible subjects.
        i (int): The index of the testing subject.
        j (int): The index of the validation subject.
        --------------------------------------------
        
        -- Returns ---------------------------------
        (list of str): A list of subjects in the training fold.
        --------------------------------------------
    """
    training_fold = []
    for k, training_subject in enumerate(subject_list):
        if (i != k) and (j != k):
            training_fold.append(training_subject)
    return training_fold
    