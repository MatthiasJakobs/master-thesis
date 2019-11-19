from ExperimentBase import Pose_JHMDB

conf = {}

conf["batch_size"] = 2
conf["learning_rate"] = 1e-5
conf["use_random"] = True
conf["validation_amount"] = 0.1
conf["limit_data_percent"] = 1
conf["numpy_seed"] = 30004
conf["project_dir"] = "refine_jhmdb"
conf["evaluate_rate"] = 1000
conf["val_batch_size"] = 1
conf["total_iterations"] = 30000

conf["num_blocks"] = 2
conf["nr_context"] = 0
conf["name"] = "without_context_2"
ft = Pose_JHMDB(conf, use_pretrained=True, pretrained="/data/mjakobs/data/jhmdb_refine/woc2", nr_aug=6)
ft.run_experiment()

conf["num_blocks"] = 4
conf["nr_context"] = 0
conf["name"] = "without_context_4"
ft = Pose_JHMDB(conf, use_pretrained=True, pretrained="/data/mjakobs/data/jhmdb_refine/woc4", nr_aug=6)
ft.run_experiment()

conf["batch_size"] = 1
conf["nr_context"] = 0
conf["num_blocks"] = 8
conf["name"] = "without_context_8"
ft = Pose_JHMDB(conf, use_pretrained=True, pretrained="/data/mjakobs/data/jhmdb_refine/woc8", nr_aug=6)
ft.run_experiment()

conf["nr_context"] = 2
conf["num_blocks"] = 2
conf["name"] = "with_context_2"
ft = Pose_JHMDB(conf, use_pretrained=True, pretrained="/data/mjakobs/data/jhmdb_refine/wic2", nr_aug=6)
ft.run_experiment()

conf["batch_size"] = 2
conf["num_blocks"] = 4
conf["name"] = "with_context_4"
ft = Pose_JHMDB(conf, use_pretrained=True, pretrained="/data/mjakobs/data/jhmdb_refine/wic4", nr_aug=6)
ft.run_experiment()

conf["batch_size"] = 1
conf["num_blocks"] = 8
conf["name"] = "with_context_8"
ft = Pose_JHMDB(conf, use_pretrained=True, pretrained="/data/mjakobs/data/jhmdb_refine/wic8", nr_aug=6)
ft.run_experiment()
