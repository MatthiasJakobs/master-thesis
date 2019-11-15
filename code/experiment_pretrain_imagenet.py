from ExperimentBase import StemImageNet

conf = {}

conf["learning_rate"] = 2e-5
conf["validation_amount"] = 0.1 # 10 percent
conf["numpy_seed"] = 30004
conf["project_dir"] = ""

conf["batch_size"] = 2
conf["total_iterations"] = 1
conf["evaluate_rate"] =  500
conf["limit_data_percent"] = 1 # limit dataset to x percent (for testing)

conf["name"] = "imagenet_test"
stem = StemImageNet(conf)
stem.run_experiment()