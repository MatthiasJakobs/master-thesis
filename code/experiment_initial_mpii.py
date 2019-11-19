from ExperimentBase import MPIIExperiment
'''
    Initial experiment to replicate Luvizon results.
'''

conf = {}

conf["learning_rate"] = 1e-3
conf["nr_epochs"] = 150
conf["validation_amount"] = 0.1 # 10 percent
conf["limit_data_percent"] = 1 # limit dataset to x percent (for testing)
conf["numpy_seed"] = 30004
conf["use_saved_tensors"] = True
conf["use_random_parameters"] = True
conf["project_dir"] = "initial_experiment_augmented_3"
conf["evaluate_rate"] = 1000
conf["augmentation_amount"] = 3

## With Context
### 2 Blocks
# conf["num_blocks"] = 2
# conf["nr_context"] = 2
# conf["name"] = "with_context_2"
# conf["batch_size"] = 45
# conf["val_batch_size"] = conf["batch_size"]
# conf["total_iterations"] = 120001
# experiment = MPIIExperiment(conf)
# experiment.run_experiment()

### 4 Blocks
# conf["num_blocks"] = 4
# conf["nr_context"] = 2
# conf["name"] = "with_context_4"
# conf["batch_size"] = 30
# conf["val_batch_size"] = conf["batch_size"]
# conf["total_iterations"] = 130001
# experiment = MPIIExperiment(conf)
# experiment.run_experiment()

### 8 Blocks
conf["num_blocks"] = 8
conf["nr_context"] = 2
conf["name"] = "with_context_8"
conf["batch_size"] = 20
conf["val_batch_size"] = conf["batch_size"]
conf["total_iterations"] = 150001
experiment = MPIIExperiment(conf)
experiment.run_experiment()

## Without Context
### 2 Blocks
conf["nr_context"] = 0
conf["num_blocks"] = 2
conf["name"] = "without_context_2"
conf["batch_size"] = 45
conf["total_iterations"] = 120001
conf["val_batch_size"] = conf["batch_size"]
experiment = MPIIExperiment(conf)
experiment.run_experiment()

### 4 Blocks
conf["num_blocks"] = 4
conf["name"] = "without_context_4"
conf["batch_size"] = 30
conf["total_iterations"] = 130001
conf["val_batch_size"] = conf["batch_size"]
experiment = MPIIExperiment(conf)
experiment.run_experiment()

### 8 Blocks
conf["num_blocks"] = 8
conf["name"] = "without_context_8"
conf["batch_size"] = 20
conf["total_iterations"] = 150001
conf["val_batch_size"] = conf["batch_size"]

experiment = MPIIExperiment(conf)
experiment.run_experiment()
