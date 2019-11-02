from ExperimentBase import Pose_Mixed

conf = {}

conf["batch_size"] = 30
conf["learning_rate"] = 1e-3
conf["use_random"] = True
conf["validation_amount"] = 0.1
conf["limit_data_percent"] = 1
conf["numpy_seed"] = 30004
conf["nr_context"] = 2
conf["project_dir"] = ""
conf["evaluate_rate"] = 5000

conf["total_iterations"] = 140000
conf["num_blocks"] = 4
conf["name"] = "pose_mixed"
ft = Pose_Mixed(conf)
ft.run_experiment()
