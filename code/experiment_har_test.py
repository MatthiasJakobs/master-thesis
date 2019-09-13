from ExperimentBase import HAR_Testing_Experiment

conf = {}

conf["batch_size"] = 7
conf["val_batch_size"] = conf["batch_size"]
conf["learning_rate"] = 2e-5
conf["nr_epochs"] = 150
conf["validation_amount"] = 0.1 # 10 percent
conf["limit_data_percent"] = 1 # limit dataset to x percent (for testing)
conf["numpy_seed"] = 30004
conf["num_blocks"] = 4
conf["name"] = "har_test"
conf["use_saved_tensors"] = True
conf["nr_context"] = 0
conf["project_dir"] = ""
conf["evaluate_rate"] = 10000

mpii = HAR_Testing_Experiment(conf)
mpii.run_experiment()
