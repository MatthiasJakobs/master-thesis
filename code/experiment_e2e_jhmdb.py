from ExperimentBase import HAR_E2E

conf = {}

conf["batch_size"] = 6
conf["val_batch_size"] = conf["batch_size"]
conf["learning_rate"] = 2e-5
conf["validation_amount"] = 0.1 # 10 percent
conf["limit_data_percent"] = 1 # limit dataset to x percent (for testing)
conf["numpy_seed"] = 30004
conf["num_blocks"] = 4
conf["nr_context"] = 0
conf["project_dir"] = ""
conf["total_iterations"] = 160000
conf["evaluate_rate"] = 5000
conf["name"] = "har_e2e_jhmdb"

har = HAR_E2E(conf)
har.run_experiment()