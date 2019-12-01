from ExperimentBase import *

conf = {}

# example values, need to be present but are not important
conf["batch_size"] = 1
conf["val_batch_size"] = conf["batch_size"]
conf["learning_rate"] = 2e-5
conf["nr_epochs"] = 150
conf["validation_amount"] = 0.1 # 10 percent
conf["numpy_seed"] = 30004
conf["use_saved_tensors"] = True
conf["project_dir"] = ""
conf["total_iterations"] = 1
conf["evaluate_rate"] = 1
conf["name"] = "eval_pose_jhmdb"


conf["limit_data_percent"] = 1 # limit dataset to x percent (for testing)

conf["nr_context"] = 0
conf["num_blocks"] = 2
model = Pose_JHMDB(conf, validate=True)
print("without context, nr_blocks = 2")
print("refined")
print(model.test(pretrained_model="/data/mjakobs/code/master-thesis/code/experiments/refine_jhmdb/without_context_2/weights/weights_00028000", refine_bounding_box=True))
print("not refined")
print(model.test(pretrained_model="/data/mjakobs/code/master-thesis/code/experiments/refine_jhmdb/without_context_2/weights/weights_00028000", refine_bounding_box=False))

conf["nr_context"] = 0
conf["num_blocks"] = 4
model = Pose_JHMDB(conf, validate=True)
print("without context, nr_blocks = 4")
print("refined")
print(model.test(pretrained_model="/data/mjakobs/code/master-thesis/code/experiments/refine_jhmdb/without_context_4/weights/weights_00028000", refine_bounding_box=True))
print("not refined")
print(model.test(pretrained_model="/data/mjakobs/code/master-thesis/code/experiments/refine_jhmdb/without_context_4/weights/weights_00028000", refine_bounding_box=False))

conf["nr_context"] = 0
conf["num_blocks"] = 8
model = Pose_JHMDB(conf, validate=True)
print("without context, nr_blocks = 8")
print("refined")
print(model.test(pretrained_model="/data/mjakobs/code/master-thesis/code/experiments/refine_jhmdb/without_context_8/weights/weights_00009000", refine_bounding_box=True))
print("not refined")
print(model.test(pretrained_model="/data/mjakobs/code/master-thesis/code/experiments/refine_jhmdb/without_context_8/weights/weights_00009000", refine_bounding_box=False))

conf["nr_context"] = 2
conf["num_blocks"] = 2
model = Pose_JHMDB(conf, validate=True)
print("with context, nr_blocks = 2")
print("refined")
print(model.test(pretrained_model="/data/mjakobs/code/master-thesis/code/experiments/refine_jhmdb/with_context_2/weights/weights_00001000", refine_bounding_box=True))
print("not refined")
print(model.test(pretrained_model="/data/mjakobs/code/master-thesis/code/experiments/refine_jhmdb/with_context_2/weights/weights_00001000", refine_bounding_box=False))

conf["nr_context"] = 2
conf["num_blocks"] = 4
model = Pose_JHMDB(conf, validate=True)
print("with context, nr_blocks = 4")
print("refined")
print(model.test(pretrained_model="/data/mjakobs/code/master-thesis/code/experiments/refine_jhmdb/with_context_4/weights/weights_00009000", refine_bounding_box=True))
print("not refined")
print(model.test(pretrained_model="/data/mjakobs/code/master-thesis/code/experiments/refine_jhmdb/with_context_4/weights/weights_00009000", refine_bounding_box=False))

conf["nr_context"] = 2
conf["num_blocks"] = 8
model = Pose_JHMDB(conf, validate=True)
print("with context, nr_blocks = 8")
print("refined")
print(model.test(pretrained_model="/data/mjakobs/code/master-thesis/code/experiments/refine_jhmdb/with_context_8/weights/weights_00005000", refine_bounding_box=True))
print("not refined")
print(model.test(pretrained_model="/data/mjakobs/code/master-thesis/code/experiments/refine_jhmdb/with_context_8/weights/weights_00005000", refine_bounding_box=False))
