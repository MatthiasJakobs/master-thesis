from ExperimentBase import MPIIExperiment
'''
    Initial experiment to replicate Luvizon results.
'''

conf = {}

conf["learning_rate"] = 1e-3
conf["validation_amount"] = 0.1 # 10 percent
conf["limit_data_percent"] = 1 # limit dataset to x percent (for testing)
conf["numpy_seed"] = 30004
conf["use_saved_tensors"] = True
conf["use_random_parameters"] = True
conf["nr_context"] = 2
conf["project_dir"] = ""
conf["evaluate_rate"] = 5000
conf["lr_milestones"] = [50000, 100000]

## With Context

## 8 Blocks
conf["num_blocks"] = 8
conf["name"] = "mpii_learningrate"
conf["batch_size"] = 20
conf["total_iterations"] = 190001
conf["val_batch_size"] = conf["batch_size"]

experiment = MPIIExperiment(conf)
experiment.run_experiment()
