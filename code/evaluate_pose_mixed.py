from ExperimentBase import Pose_Mixed

conf = {}

conf["batch_size"] = 30
conf["learning_rate"] = 1e-3
conf["use_random"] = True
conf["validation_amount"] = 0.1
conf["limit_data_percent"] = 1
conf["numpy_seed"] = 30004
conf["project_dir"] = ""
conf["evaluate_rate"] = 1000

conf["total_iterations"] = 35000
conf["num_blocks"] = 4
conf["nr_context"] = 2
conf["name"] = "eval_pose_mixed"
ft = Pose_Mixed(conf)
ft.test(pretrained_model="/data/mjakobs/data/pose_mixed_final")
