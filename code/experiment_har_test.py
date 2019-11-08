from ExperimentBase import HAR_Testing_Experiment

conf = {}

conf["learning_rate"] = 2e-5
conf["validation_amount"] = 0.1 # 10 percent
conf["numpy_seed"] = 30004
conf["num_blocks"] = 4
conf["nr_context"] = 2
conf["project_dir"] = "har_initial"

conf["batch_size"] = 20
conf["val_batch_size"] = 1
conf["total_iterations"] = 10000
conf["evaluate_rate"] =  500
conf["limit_data_percent"] = 1 # limit dataset to x percent (for testing)

# conf["batch_size"] = 6
#conf["val_batch_size"] = conf["batch_size"]
# conf["total_iterations"] = 160000
# conf["evaluate_rate"] = 5000
# conf["limit_data_percent"] = 1 # limit dataset to x percent (for testing)

conf["name"] = "no_finetune"
conf["fine_tune"] = False
conf["use_timedistributed"] = False
har = HAR_Testing_Experiment(conf)
har.run_experiment()

# conf["name"] = "with_finetune"
# conf["start_finetuning"] = 1500
# conf["fine_tune"] = True
# conf["lr_milestones"] = [1500, 6000]
# har = HAR_Testing_Experiment(conf)
# har.run_experiment()
