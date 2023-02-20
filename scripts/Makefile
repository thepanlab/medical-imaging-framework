j ?=
args ?=

define find.functions
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'
endef

help:
	@echo '----------------------------'
	@echo 'The following j exist:'
	@echo '----------------------------'
	@echo 'j: The default value is none. This should be something like j="my_config.json", if applicable.'
	@echo ''
	@echo '-----------------------------------'
	@echo 'The following commands can be used:'
	@echo '-----------------------------------'
	$(call find.functions)
	@echo ''


confusion_matrix: ## Creates a single confusion matrix for the specified inputs.
confusion_matrix: 
	python3 -W ignore -m results_processing.confusion_matrix.confusion_matrix -j ${j}


confusion_matrix_many: ## Creates a confusion matrix for a every item in a given data folder.
confusion_matrix_many:
	python3 -W ignore -m results_processing.confusion_matrix.confusion_matrix_many -j ${j}


confusion_matrix_many_means: ## Creates the means of confusion matrix for a every config and subject.
confusion_matrix_many_means:
	python3 -W ignore -m results_processing.confusion_matrix.confusion_matrix_many_means -j ${j}


epoch_counting: ## Counts the number of epochs within a particular data folder.
epoch_counting: 
	python3 -W ignore -m results_processing.epoch_counting.epoch_counting -j ${j}


grad_cam: ## Creates heatmap images from a specified item or data folder.
grad_cam: 
	python3 -W ignore -m results_processing.grad_cam.grad_cam -j ${j}


grad_cam_many: ## Creates heatmap images from a specified data folder.
grad_cam_many: 
	python3 -W ignore -m results_processing.grad_cam.grad_cam_many -j ${j}


learning_curve: ## Creates a learning curve from the specified data.
learning_curve: 
	python3 -W ignore -m results_processing.learning_curve.learning_curve -j ${j}


learning_curve_many: ## Creates many learning curves from the specified data folder.
learning_curve_many: 
	python3 -W ignore -m results_processing.learning_curve.learning_curve_many -j ${j}


metrics_table: ## Creates a metrics_table for the given data folder.
metrics_table: 
	python3 -W ignore -m results_processing.metrics_table.metrics_table -j ${j}


prediction: ## Creates predictions from a given set of inputs.
prediction: 
	python3 -W ignore -m results_processing.prediction.prediction -j ${j}


roc_curve: ## Creates a ROC curve from the specified data.
roc_curve: 
	python3 -W ignore -m results_processing.roc_curve.roc_curve -j ${j}


roc_curve_many: ## Creates many ROC curves from the specified data folder.
roc_curve_many: 
	python3 -W ignore -m results_processing.roc_curve.roc_curve_many -j ${j}


summary_table: ## Creates a summary table from the specified data folder.
summary_table: 
	python3 -W ignore -m results_processing.summary_table.summary_table -j ${j}


tabled_prediction_info: ## Creates a dataframe of existing prediction results.
tabled_prediction_info: 
	python3 -W ignore -m results_processing.tabled_prediction_info.tabled_prediction_info -j ${j}


## -------------------------------------------------------------------------- ##


training_inner_loop: ## Runs the inner loop training program. Specify arguments with "args='--file X'" or "args='--folder X'"
training_inner_loop: 
	python3 -W ignore -m training.training_inner_loop $(args)

clean: clean-build clean-pyc clean-test

clean-build:
	rm -fr build/
	rm -fr dist/
	rm -fr .eggs/
	find . -name '*.egg-info' -exec rm -fr {} +
	find . -name '*.egg' -exec rm -f {} +

clean-pyc:
	find . -name '*.pyc' -exec rm -f {} +
	find . -name '*.pyo' -exec rm -f {} +
	find . -name '*~' -exec rm -f {} +
	find . -name '__pycache__' -exec rm -fr {} +

clean-test:
	rm -fr .tox/
	rm -f .coverage
	rm -fr htmlcov/

lint:
	flake8 leviathan_serving tests

test:
	python setup.py test

test-all:
	tox

coverage:
	coverage run --source leviathan_serving setup.py test
	coverage report -m
	coverage html
	$(BROWSER) htmlcov/index.html

docs:
	rm -f docs/leviathan_serving.rst
	rm -f docs/modules.rst
	sphinx-apidoc -o docs/ leviathan_serving
	$(MAKE) -C docs clean
	$(MAKE) -C docs html
	$(BROWSER) docs/_build/html/index.html

servedocs: docs
	watchmedo shell-command -p '*.rst' -c '$(MAKE) -C docs html' -R -D .

release: clean
	python setup.py sdist upload
	python setup.py bdist_wheel upload

dist: clean
	python setup.py sdist
	python setup.py bdist_wheel
	ls -l dist

install: clean
	python setup.py install