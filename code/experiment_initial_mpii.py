from ExperimentBase import MPIIExperiment
'''
    Initial experiment to replicate Luvizon results.
'''

conf = {}

conf["learning_rate"] = 1e-3
conf["nr_epochs"] = 150
conf["total_iterations"] = 100000
conf["validation_amount"] = 0.1 # 10 percent
conf["limit_data_percent"] = 1 # limit dataset to x percent (for testing)
conf["numpy_seed"] = 30004
conf["use_saved_tensors"] = True
conf["use_random_parameters"] = True
conf["nr_context"] = 0
conf["project_dir"] = "initial_experiment"
conf["evaluate_rate"] = 1000

## With Context
### 2 Blocks
# conf["num_blocks"] = 2
# conf["name"] = "with_context_2"
# conf["batch_size"] = 45
# conf["val_batch_size"] = conf["batch_size"]
# run_experiment_mpii(conf)

### 4 Blocks
# conf["num_blocks"] = 4
# conf["name"] = "with_context_4"
# conf["batch_size"] = 35
# conf["val_batch_size"] = conf["batch_size"]
# run_experiment_mpii(conf)

### 8 Blocks
# conf["num_blocks"] = 8
# conf["name"] = "with_context_8"
# conf["batch_size"] = 25
# conf["val_batch_size"] = conf["batch_size"]
# run_experiment_mpii(conf)

## Without Context
### 2 Blocks
conf["nr_context"] = 0
conf["num_blocks"] = 2
conf["name"] = "without_context_2"
conf["batch_size"] = 45
conf["total_iterations"] = 100000
conf["val_batch_size"] = conf["batch_size"]
experiment = MPIIExperiment(conf)
experiment.run_experiment()

### 4 Blocks
conf["num_blocks"] = 4
conf["name"] = "without_context_4"
conf["batch_size"] = 30
conf["total_iterations"] = 130000
conf["val_batch_size"] = conf["batch_size"]
experiment = MPIIExperiment(conf)
experiment.run_experiment()

### 8 Blocks
conf["num_blocks"] = 8
conf["name"] = "without_context_8"
conf["batch_size"] = 20
conf["total_iterations"] = 185000
conf["val_batch_size"] = conf["batch_size"]

experiment = MPIIExperiment(conf)
experiment.run_experiment()
