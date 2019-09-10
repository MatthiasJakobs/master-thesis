from ExperimentBase import Finetune_JHMDB

conf = {}

conf["batch_size"] = 1
conf["learning_rate"] = 1e-15
conf["nr_epochs"] = 100
conf["validation_amount"] = 0.1
conf["limit_data_percent"] = 1
conf["numpy_seed"] = 30004
conf["name"] = "fine_tuning_jhmdb"
conf["num_blocks"] = 4
conf["nr_context"] = 0
conf["project_dir"] = ""
conf["evaluate_rate"] = 50

ft = Finetune_JHMDB(conf)
ft.run_experiment()
