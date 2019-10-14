from ExperimentBase import MPIIExperiment
'''
    Initial experiment to replicate Luvizon results.
'''

conf = {}

conf["learning_rate"] = 1e-3
conf["validation_amount"] = 0.1 # 10 percent
conf["limit_data_percent"] =1 # limit dataset to x percent (for testing)
conf["numpy_seed"] = 30004
conf["batch_size"] = 45
conf["use_saved_tensors"] = True
conf["use_random_parameters"] = True
conf["nr_context"] = 0
conf["project_dir"] = ""
conf["evaluate_rate"] = 20000

## Without Context
### 2 Blocks
conf["nr_context"] = 0
conf["num_blocks"] = 2
conf["name"] = "mpii_3_augmentation"
conf["total_iterations"] = 120000
conf["val_batch_size"] = conf["batch_size"]
conf["augmentation_amount"] = 3
experiment = MPIIExperiment(conf)
experiment.run_experiment()
