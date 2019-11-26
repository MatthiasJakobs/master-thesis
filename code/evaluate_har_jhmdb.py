from ExperimentBase import *

conf = {}

# example values, need to be present but are not important
conf["batch_size"] = 2
conf["val_batch_size"] = conf["batch_size"]
conf["learning_rate"] = 2e-5
conf["nr_epochs"] = 150
conf["validation_amount"] = 0.1 # 10 percent
conf["numpy_seed"] = 30004
conf["num_blocks"] = 4
conf["use_saved_tensors"] = True
conf["nr_context"] = 0
conf["project_dir"] = ""
conf["total_iterations"] = 3
conf["evaluate_rate"] = 3
conf["name"] = "eval"


conf["limit_data_percent"] = 1 # limit dataset to x percent (for testing)

har = HAR_Testing_Experiment(conf, validate=True)
print(har.test(pretrained_model="/data/mjakobs/code/master-thesis/code/experiments/har_initial/no_finetune_with_td_refined_smaller_lr/weights/weights_00003000"))
