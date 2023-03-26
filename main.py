import read
import display_data
import train

TRAIN, TEST, META = read.get_paths()

test_filenames, test_labels, test_data = read.read(TEST)
train_filenames, train_labels, train_data = read.read(TRAIN)
label_names = read.read_labels(META)

train_filenames, train_labels, train_data, valid_filenames, valid_labels, valid_data = read.split_validation(train_filenames, train_labels, train_data)

display_data.data_distribution(train_labels)
display_data.show_images(train_data, train_labels, label_names)

history, model = train.train(train_data, train_labels, valid_data, valid_labels)
train.evaluate_model(test_data, test_labels, model)