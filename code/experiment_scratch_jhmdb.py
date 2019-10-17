from ExperimentBase import Pose_JHMDB

conf = {}

conf["batch_size"] = 2
conf["learning_rate"] = 1e-3
conf["use_random"] = True
conf["validation_amount"] = 0.1
conf["limit_data_percent"] = 1
conf["numpy_seed"] = 30004
conf["nr_context"] = 0
conf["project_dir"] = "scratch_jhmdb"
conf["evaluate_rate"] = 10000

conf["total_iterations"] = 130000
conf["num_blocks"] = 2
conf["name"] = "without_context_2"
ft = Pose_JHMDB(conf, use_pretrained=False, nr_aug=1)
ft.run_experiment()

conf["total_iterations"] = 160000
conf["num_blocks"] = 4
conf["name"] = "without_context_4"
ft = Pose_JHMDB(conf, use_pretrained=False, nr_aug=1)
ft.run_experiment()

conf["batch_size"] = 1
conf["total_iterations"] = 190000
conf["num_blocks"] = 8
conf["name"] = "without_context_8"
ft = Pose_JHMDB(conf, use_pretrained=False, nr_aug=1)
ft.run_experiment()

conf["nr_context"] = 2
conf["total_iterations"] = 130000
conf["num_blocks"] = 2
conf["name"] = "with_context_2"
ft = Pose_JHMDB(conf, use_pretrained=False, nr_aug=1)
ft.run_experiment()

conf["total_iterations"] = 160000
conf["num_blocks"] = 4
conf["name"] = "with_context_4"
ft = Pose_JHMDB(conf, use_pretrained=False, nr_aug=1)
ft.run_experiment()

conf["total_iterations"] = 2200000
conf["batch_size"] = 1
conf["num_blocks"] = 8
conf["name"] = "with_context_8"
ft = Pose_JHMDB(conf, use_pretrained=False, nr_aug=1)
ft.run_experiment()
