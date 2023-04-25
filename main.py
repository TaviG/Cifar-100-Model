import read
import display_data
import train
import time

TRAIN, TEST, META = read.get_paths()

train_loader, valid_loader, test_loader = read.read2(valid_size = 0.1, batch_size = 32, num_workers = 0)
display_data.data_distribution2(train_loader)

test_filenames, test_labels, test_data = read.read(TEST)
train_filenames, train_labels, train_data = read.read(TRAIN)
label_names = read.read_labels(META)
display_data.show_images2(train_loader, label_names)
train_filenames, train_labels, train_data, valid_filenames, valid_labels, valid_data = read.split_validation(train_filenames, train_labels, train_data)

display_data.data_distribution(train_labels)
display_data.show_images(train_data, train_labels, label_names)

#history, model = train.train(train_data, train_labels, valid_data, valid_labels)
# model = train.train2()
model, clf = train.train(train_data, train_labels, valid_data, valid_labels)
start_time = time.time()
acc = train.evaluate_model(model, test_data, test_labels, label_names, clf)

print("Accuracy: ",acc, "%" )
print("--- %s seconds ---" % (time.time() - start_time))