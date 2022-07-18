.. sectnum::

=====
Title
=====


.. contents:: Table of Contents
    :depth: 2

------------------------
``scripts/get_means.py``
------------------------

Assumptions
============

#. The Structure of the file is like this e.g.
    *	Directory:

        * S1_category1
        * S1_category2
        * S2_category1
        * …
#. The file name of the image should contain at least the subject and the label name.e.g. ``125_S1_category1.png``. The logic of the function only works if this information is provided in the file name. different information that is contained in the filename should be separated by underscore(``_``)
#. The data is divided by n subjects. The division is performed by subject because we observed that is variability among the subjects and we want to replicate how the model would behave with a new subject.
#. Function reads the images as grayscale.
#. The images have all the same shape

Description
============

* The splitting will be done according to the subject.
* Syntax to label subject: You will enter the first letter, e.g. ``S``. This will give idea how to look for subject. The code will look for ``S1``, ``S2`` and so on. It will detect the total number of subjects, the variable n is used.
* For Cross validation: it will calculate the train mean for each of the folds(n values) and well as the train-validation whole dataset (1 value).
* For Nested cross validation: it will calculate the train for each of the folds in inner loop (n time n-1 values) as wells as outer loop (n values).
* Config file (``json`` file):
    * ``files_directory``: one specific address for files
    * ``letter``: letter that assigned the subject
    * ``output_means``: output address where to store mean values as pandas dataframe

How to use it
=============
Run it like, e.g.:

.. code-block:: bash

    python get_means.py config_means.json