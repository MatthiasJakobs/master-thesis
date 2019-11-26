from ExperimentBase import HAR_Testing_Experiment

conf = {}

conf["learning_rate"] = 1e-6
conf["validation_amount"] = 0.1 # 10 percent
conf["numpy_seed"] = 30004
conf["num_blocks"] = 4
conf["nr_context"] = 2
conf["project_dir"] = "har_initial"

conf["batch_size"] = 12
conf["val_batch_size"] = 1
conf["total_iterations"] = 200000
conf["evaluate_rate"] =  1000
conf["limit_data_percent"] = 1 # limit dataset to x percent (for testing)

# conf["batch_size"] = 6
#conf["val_batch_size"] = conf["batch_size"]
# conf["total_iterations"] = 160000
# conf["evaluate_rate"] = 5000
# conf["limit_data_percent"] = 1 # limit dataset to x percent (for testing)

# conf["learning_rate"] = 1e-6
# conf["batch_size"] = 12
# conf["name"] = "no_finetune_with_td_refined_smaller_lr"
# conf["fine_tune"] = False
# conf["use_timedistributed"] = True
# har = HAR_Testing_Experiment(conf, start_at="/data/mjakobs/data/har_jhmdb_12000")
# har.run_experiment()

# conf["learning_rate"] = 1e-5
# conf["batch_size"] = 2
# conf["name"] = "with_finetune_with_td_refined"
# conf["fine_tune"] = True
# conf["use_timedistributed"] = True
# har = HAR_Testing_Experiment(conf, start_at="/data/mjakobs/code/master-thesis/code/experiments/har_initial/no_finetune_with_td_refined_smaller_lr/weights/weights_00003000")
# har.run_experiment()

# conf["learning_rate"] = 1e-6
# conf["batch_size"] = 1
# conf["name"] = "with_finetune_with_td_refined_smaller_lr"
# conf["fine_tune"] = True
# conf["use_timedistributed"] = True
# har = HAR_Testing_Experiment(conf, start_at="/data/mjakobs/code/master-thesis/code/experiments/har_initial/no_finetune_with_td_refined_smaller_lr/weights/weights_00003000")
# har.run_experiment()

conf["learning_rate"] = 1e-6
conf["batch_size"] = 2
conf["name"] = "with_finetune_with_td_refined_smaller_lr_bs2"
conf["fine_tune"] = True
conf["use_timedistributed"] = True
har = HAR_Testing_Experiment(conf, start_at="/data/mjakobs/code/master-thesis/code/experiments/har_initial/no_finetune_with_td_refined_smaller_lr/weights/weights_00003000")
har.run_experiment()

# conf["name"] = "finetune_with_td"
# conf["fine_tune"] = True
# conf["start_finetuning"] = 0
# conf["batch_size"] = 12
# conf["learning_rate"] = 1e-6
# conf["use_timedistributed"] = True
# har = HAR_Testing_Experiment(conf, pretrained_model="/data/mjakobs/data/har_jhmdb_5500")
# har.run_experiment()

# conf["name"] = "with_finetune"
# conf["start_finetuning"] = 1500
# conf["fine_tune"] = True
# conf["lr_milestones"] = [1500, 6000]
# har = HAR_Testing_Experiment(conf)
# har.run_experiment()
