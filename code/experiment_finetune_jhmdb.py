from ExperimentBase import Pose_JHMDB

conf = {}

conf["batch_size"] = 2
conf["learning_rate"] = 1e-7
conf["nr_epochs"] = 100
conf["validation_amount"] = 0.1
conf["limit_data_percent"] = 1
conf["numpy_seed"] = 30004
conf["name"] = "fine_tuning_jhmdb"
conf["num_blocks"] = 4
conf["nr_context"] = 0
conf["project_dir"] = ""
conf["evaluate_rate"] = 1000

ft = Pose_JHMDB(conf, use_pretrained=True)
ft.run_experiment()
