from random import shuffle

def generate_pairs(test_subject_list, validation_subject_list, subject_list, do_shuffle):
    """ Generates subject-subject pairs

        -- Input Parameters ------------------------
        test_subject_list (list of str): List of test subjects.
        validation_subject_list (list of str): List of validation subjects.
        validation_subject_list (list of str): List of all subjects.
        do_shuffle (bool): If the fold list should be shuffled or not.
        --------------------------------------------
        
        -- Returns ---------------------------------
        (list of str tuples): A list of subject pairs.
        --------------------------------------------
    """
    # Generate subject-subject tuples
    folds = []
    for test_subject in test_subject_list:
        
        # Outer loop: use test subjects only
        if validation_subject_list is None:   
            folds.extend(_get_training_combos(subject_list, test_subject))
            
        # Inner loop: use validation subjects
        else:   
            folds.extend(_get_training_combos(validation_subject_list, test_subject))
        
    # Shuffle the folds
    if do_shuffle:
        shuffle(folds)
    return folds
    


def generate_folds(test_subject_list, validation_subject_list, subject_list, test_subject, do_shuffle, training_subject=None):
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
            'training': _fill_training_fold(subject_list, test_subject, test_subject)
        }]
        
    # If inner loop, get the test-val combinations.
    else:        
        folds = []
        if training_subject:    
            folds.append({
                'testing': [test_subject],
                'training': _fill_training_fold(validation_subject_list, test_subject, training_subject), 
                'validation': [training_subject]
            })
        
        else:
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
        test_subject (int): The testing subject.
        subject (int): The paired validation/testing subject.
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


def _get_training_combos(subject_list, test_subject):
    """ Fills the training fold for some subject.

        -- Input Parameters ------------------------
        subject_list (list of str): A list of possible subjects.
        test_subject (int): The the testing subject.
        --------------------------------------------
        
        -- Returns ---------------------------------
        (list of str tuples): A list of subject pairs.
        --------------------------------------------
    """
    folds = []
    for subject in subject_list:
        if subject != test_subject:
            folds.append((test_subject, subject))
    return folds
    
