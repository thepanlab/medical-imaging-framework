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
    # If outer loop, compare the test subjects.
    if validation_subject_list is None:        
        folds = [{
            'testing': [test_subject],
            'training': _fill_training_fold(test_subject_list, test_subject, test_subject)
        }]
        
    # If inner loop, get the test-val combinations.
    else:        
        folds = []
        for subject in validation_subject_list:
            if subject != test_subject:     
                folds.append({
                    'testing': [test_subject],
                    'training': _fill_training_fold(validation_subject_list, test_subject, subject), 
                    'validation': [subject]
                })
        
    # Shuffle the data and get the number of training rotations
    if do_shuffle:
        shuffle(folds)
    return folds, len(folds)


def _fill_training_fold(subject_list, test_subject, subject):
    """ Fills the training fold for some subject.

        -- Input Parameters ------------------------
        subject_list (list of str): A list of possible subjects.
        test_subject (int): The index of the testing subject.
        subject (int): The index of the paired validation/testing subject.
        --------------------------------------------
        
        -- Returns ---------------------------------
        (list of str): A list of subjects in the training fold.
        --------------------------------------------
    """
    training_fold = []
    for s in subject_list:
        if s not in [test_subject, subject]:
            training_fold.append(s)
    return training_fold
    
