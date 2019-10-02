from ExperimentBase import Pose_JHMDB

conf = {}

conf["batch_size"] = 2
conf["learning_rate"] = 1e-3
conf["use_random"] = True
conf["validation_amount"] = 0.1
conf["limit_data_percent"] = 1
conf["numpy_seed"] = 30004
conf["num_blocks"] = 4
conf["nr_context"] = 0
conf["project_dir"] = ""
conf["total_iterations"] = 160000
conf["evaluate_rate"] = 5000


conf["name"] = "scratch_jhmdb_1_aug_with_flip"
ft = Pose_JHMDB(conf, use_pretrained=False, nr_aug=1)
ft.run_experiment()
