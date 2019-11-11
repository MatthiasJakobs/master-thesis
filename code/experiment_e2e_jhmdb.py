from ExperimentBase import HAR_E2E

conf = {}

conf["batch_size"] = 1
conf["val_batch_size"] = 1
conf["learning_rate"] = 1e-5
conf["validation_amount"] = 0.1 # 10 percent
conf["limit_data_percent"] = 1 # limit dataset to x percent (for testing)
conf["numpy_seed"] = 30004
conf["num_blocks"] = 4
conf["nr_context"] = 2
conf["project_dir"] = ""
conf["total_iterations"] = 50000
conf["evaluate_rate"] = 1000
conf["name"] = "har_e2e_jhmdb"
conf["use_gt_bb"] = True
conf["use_timedistributed"] = True

har = HAR_E2E(conf)
har.run_experiment()
