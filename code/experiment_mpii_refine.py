from ExperimentBase import MPIIExperiment
'''
    Initial experiment to replicate Luvizon results.
'''

conf = {}

conf["learning_rate"] = 1e-4
conf["nr_epochs"] = 150
conf["validation_amount"] = 0.1 # 10 percent
conf["limit_data_percent"] = 1 # limit dataset to x percent (for testing)
conf["numpy_seed"] = 30004
conf["use_saved_tensors"] = True
conf["use_random_parameters"] = True
conf["project_dir"] = "mpii_refine"
conf["evaluate_rate"] = 500
conf["augmentation_amount"] = 3

## With Context
### 2 Blocks
conf["num_blocks"] = 2
conf["nr_context"] = 2
conf["name"] = "with_context_2"
conf["batch_size"] = 45
conf["val_batch_size"] = conf["batch_size"]
experiment = MPIIExperiment(conf, pretrained_model="/data/mjakobs/mpii_refine/wc2")
experiment.run_experiment()

### 4 Blocks
conf["num_blocks"] = 4
conf["nr_context"] = 2
conf["name"] = "with_context_4"
conf["batch_size"] = 30
conf["val_batch_size"] = conf["batch_size"]
experiment = MPIIExperiment(conf, pretrained_model="/data/mjakobs/mpii_refine/wc4")
experiment.run_experiment()

### 8 Blocks
conf["num_blocks"] = 8
conf["nr_context"] = 2
conf["name"] = "with_context_8"
conf["batch_size"] = 20
conf["val_batch_size"] = conf["batch_size"]
experiment = MPIIExperiment(conf, pretrained_model="/data/mjakobs/mpii_refine/wc8")
experiment.run_experiment()

## Without Context
### 2 Blocks
conf["nr_context"] = 0
conf["num_blocks"] = 2
conf["name"] = "without_context_2"
conf["batch_size"] = 45
conf["val_batch_size"] = conf["batch_size"]
experiment = MPIIExperiment(conf, pretrained_model="/data/mjakobs/mpii_refine/wo2")
experiment.run_experiment()

### 4 Blocks
conf["num_blocks"] = 4
conf["name"] = "without_context_4"
conf["batch_size"] = 30
conf["val_batch_size"] = conf["batch_size"]
experiment = MPIIExperiment(conf, pretrained_model="/data/mjakobs/mpii_refine/wo4")
experiment.run_experiment()

### 8 Blocks
conf["num_blocks"] = 8
conf["name"] = "without_context_8"
conf["batch_size"] = 20
conf["val_batch_size"] = conf["batch_size"]
experiment = MPIIExperiment(conf, pretrained_model="/data/mjakobs/mpii_refine/wo8")
experiment.run_experiment()
