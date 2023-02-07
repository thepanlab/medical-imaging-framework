from random import shuffle


def generate_folds(subject_list, test_subject, in_rotations, do_shuffle):
    """ Generates folds for the subject.
        
        -- Input Parameters ------------------------
        subject_list (list of str): A list of subject names.
        test_subject (str): The current test subject name.
        in_rotations (int or 'all'): How many rotations were specified in the configuration.
        do_shuffle (bool): If the fold list should be shuffled or not.
        --------------------------------------------
        
        -- Returns ---------------------------------
        (list of dict): A list of folds, containing the subjects for testing, validation, and training.
        (int): The number of rotations for the training loop.
        --------------------------------------------
    """
    folds = []
    # For this test subject, find all combinations for the testing data
    i = subject_list.index(test_subject)
    for j, validation_subject in enumerate(subject_list):
        if i != j:
            subject_fold = {'training': _fill_training_fold(subject_list, i, j), 'validation': [validation_subject], 'testing': [test_subject]}
            folds.append(subject_fold)
    
    # Shuffle the data and get the number of training rotations
    if do_shuffle:
        shuffle(folds)
    if (in_rotations == 'all') or (in_rotations > len(folds)):
        return folds, len(folds)
    return folds, in_rotations


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
    