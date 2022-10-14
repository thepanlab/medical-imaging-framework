OPTIONS ?=

define find.functions
	@fgrep -h "##" $(MAKEFILE_LIST) | fgrep -v fgrep | sed -e 's/\\$$//' | sed -e 's/##//'
endef

help:
	@echo '----------------------------'
	@echo 'The following options exist:'
	@echo '----------------------------'
	@echo 'OPTIONS: The default value is none. This should be something like "-j my_config.json", if applicable.'
	@echo ''
	@echo '-----------------------------------'
	@echo 'The following commands can be used:'
	@echo '-----------------------------------'
	$(call find.functions)
	@echo ''


confusion_matrix: ## Creates a single confusion matrix for the specified inputs.
confusion_matrix: 
	cd results_processing/confusion_matrix && python3 -m results_processing.confusion_matrix.confusion_matrix ${OPTIONS}


confusion_matrix_many: ## Creates a confusion matrix for a every item in a given data folder.
confusion_matrix_many:
	cd results_processing/confusion_matrix && python3 -m results_processing.confusion_matrix.confusion_matrix_many ${OPTIONS}


confusion_matrix_many_means: ## Creates the means of confusion matrix for a every config and subject.
confusion_matrix_many_means:
	cd results_processing/confusion_matrix && python3 -m results_processing.confusion_matrix.confusion_matrix_many_means ${OPTIONS}


epoch_counting: ## Counts the number of epochs within a particular data folder.
epoch_counting: 
	cd results_processing/epoch_counting && python3 -m results_processing.epoch_counting.epoch_counting ${OPTIONS}


grad_cam: ## Creates heatmap images from a specified item or data folder.
grad_cam: 
	cd results_processing/grad_cam && python3 -m results_processing.grad_cam.grad_cam ${OPTIONS}


learning_curve: ## Creates a learning curve from the specified data.
learning_curve: 
	cd results_processing/learning_curve && python3 -m results_processing.learning_curve.learning_curve ${OPTIONS}


learning_curve_graphing: ## Creates many learning curves from the specified data folder.
learning_curve_graphing: 
	cd results_processing/learning_curve && python3 -m results_processing.learning_curve.learning_curve_graphing ${OPTIONS}


roc_curve: ## Creates a ROC curve from the specified data.
roc_curve: 
	cd results_processing/roc_curve && python3 -m results_processing.roc_curve.roc_curve ${OPTIONS}


roc_curve_graphing: ## Creates many ROC curves from the specified data folder.
roc_curve_graphing: 
	cd results_processing/roc_curve && python3 -m results_processing.roc_curve.roc_curve_graphing ${OPTIONS}


summary_table: ## Creates a summary table from the specified data folder.
summary_table: 
	cd results_processing/summary_table && python3 -m results_processing.summary_table.summary_table ${OPTIONS}

