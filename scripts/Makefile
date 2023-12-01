## --- Variables and Functions -------------------------------------------------------------------- ##
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

## --- Random Search ----------------------------------------------- ##
create_random_json:
create_random_json:
	python3 -W ignore -m training.random_search.create_random_json -j ${j}


## --- Sequential Training ------------------------------------------------------------------------ ##
training_inner_loop: ## Runs the inner loop training program. Specify arguments with "args='--file X'" or "args='--folder X'"
training_inner_loop: 
	CUDA_VISIBLE_DEVICES=0 python3 -W ignore -m training.training_sequential.loop_inner.training_inner_loop $(args)


training_outer_loop: ## Runs the outer loop training program. Specify arguments with "args='--file X'" or "args='--folder X'"
training_outer_loop: 
	CUDA_VISIBLE_DEVICES=1 python3 -W ignore -m training.training_sequential.loop_outer.training_outer_loop $(args)




## --- Distributed Training ----------------------------------------------------------------------- ##
mpi_init: ## Prints a console command for running MPI based on your config parameters.
mpi_init: 
	python3 -W ignore -m training.training_multiprocessing.mpi_init $(args)


distributed_training_inner_loop: ## Runs the inner loop training program. Specify arguments with "args='--file X'" or "args='--folder X'"
distributed_training_inner_loop: 
	CUDA_VISIBLE_DEVICES=0,1 mpirun -n 5 python3 -W ignore -m training.training_multiprocessing.loop_inner.multiprocessed_training_inner_loop


distributed_training_outer_loop: ## Runs the outer loop training program. Specify arguments with "args='--file X'" or "args='--folder X'"
distributed_training_outer_loop: 
	CUDA_VISIBLE_DEVICES=0,1 mpirun -n 5 python3 -W ignore -m training.training_multiprocessing.loop_outer.multiprocessed_training_outer_loop




## --- Results Processing ------------------------------------------------------------------------- ##
predicted_formatter: ## Formats prediction values of some data.
predicted_formatter: 
	python3 -W ignore -m util.predicted_formatter.predicted_formatter


truth_formatter: ## Formats truth values of some data.
truth_formatter: 
	python3 -W ignore -m util.truth_formatter.truth_formatter


class_recall: ## Creates a list of class accuracies for every test-val fold.
class_recall: 
	python3 -W ignore -m results_processing.class_recall.class_recall -j ${j}


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


roc_curve_many_means: ## Creates ROC mean-curves from the specified data folder.
roc_curve_many_means: 
	python3 -W ignore -m results_processing.roc_curve.roc_curve_many_means -j ${j}


summary_table: ## Creates a summary table from the specified data folder.
summary_table: 
	python3 -W ignore -m results_processing.summary_table.summary_table -j ${j}


tabled_prediction_info: ## Creates a dataframe of existing prediction results.
tabled_prediction_info: 
	python3 -W ignore -m results_processing.tabled_prediction_info.tabled_prediction_info -j ${j}




## --- Custom Commands ---------------------------------------------------------------------------- ##
# Place custom commands here, if needed
