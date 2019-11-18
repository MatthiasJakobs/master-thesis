from ExperimentBase import HAR_PennAction

conf = {}

conf["validation_amount"] = 0.1 # 10 percent
conf["numpy_seed"] = 30004
conf["num_blocks"] = 4
conf["nr_context"] = 2
conf["project_dir"] = "har_pennaction_determine_lr"

conf["batch_size"] = 10
conf["val_batch_size"] = 1
conf["total_iterations"] = 25000
conf["evaluate_rate"] =  500
conf["limit_data_percent"] = 1 # limit dataset to x percent (for testing)
conf["use_gt_bb"] = True
conf["use_gt_pose"] = False
conf["use_timedistributed"] = True
conf["fine_tune"] = False 

# conf["learning_rate"] = 1e-3
# conf["name"] = "1e-3"
# har = HAR_PennAction(conf)
# har.run_experiment()

conf["learning_rate"] = 1e-4
conf["name"] = "1e-4"
har = HAR_PennAction(conf)
har.run_experiment()

conf["learning_rate"] = 1e-5
conf["name"] = "1e-5"
har = HAR_PennAction(conf)
har.run_experiment()

conf["learning_rate"] = 1e-6
conf["name"] = "1e-6"
har = HAR_PennAction(conf)
har.run_experiment()
