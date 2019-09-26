from datasets.JHMDBDataset import JHMDBDataset

ds_train = JHMDBDataset("/data/mjakobs/data/jhmdb/", train=True, val=False, use_random_parameters=False)
ds_val = JHMDBDataset("/data/mjakobs/data/jhmdb/", train=True, val=True, use_random_parameters=False)

print("train length", len(ds_train))
print("val length", len(ds_val))

train_classes = {}
val_classes = {}

for i in range(len(ds_train)):
    entry = ds_train[i]
    action = entry["action_label"]
    if not action in train_classes:
        train_classes[action] = 1.0 / len(ds_train)
    else:
        train_classes[action] = train_classes[action] + 1.0 / len(ds_train)

for i in range(len(ds_val)):
    entry = ds_val[i]
    action = entry["action_label"]
    if not action in val_classes:
        val_classes[action] = 1.0 / len(ds_val)
    else:
        val_classes[action] = val_classes[action] + 1.0 / len(ds_val)

for action in ds_train.classes:
    print(action, "train/val", train_classes[action], val_classes[action])
