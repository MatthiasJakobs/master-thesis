from ExperimentBase import HAR_Testing_Experiment

conf = {}

conf["learning_rate"] = 2e-5
conf["validation_amount"] = 0.1 # 10 percent
conf["numpy_seed"] = 30004
conf["num_blocks"] = 4
conf["nr_context"] = 2
conf["project_dir"] = "har_indepth"

conf["batch_size"] = 20
conf["val_batch_size"] = 1
conf["total_iterations"] = 90000
conf["evaluate_rate"] =  1000
conf["limit_data_percent"] = 1 # limit dataset to x percent (for testing)
conf["fine_tune"] = False # otherwise, using gt pose would make no sense

conf["use_gt_bb"] = True
conf["use_gt_pose"] = False
conf["name"] = "without_gt_pose_with_gt_bb"
har = HAR_Testing_Experiment(conf)
har.run_experiment()

conf["use_gt_bb"] = False
conf["use_gt_pose"] = False
conf["name"] = "without_gt_pose_without_gt_bb"
har = HAR_Testing_Experiment(conf)
har.run_experiment()

conf["use_gt_bb"] = True
conf["use_gt_pose"] = True
conf["name"] = "with_gt_pose_with_gt_bb"
har = HAR_Testing_Experiment(conf)
har.run_experiment()

conf["use_gt_bb"] = False
conf["use_gt_pose"] = True
conf["name"] = "with_gt_pose_without_gt_bb"
har = HAR_Testing_Experiment(conf)
har.run_experiment()