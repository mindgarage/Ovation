from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.layers import Conv1D
from keras.layers import MaxPooling1D
from datasets import TwitterEmotion

# setup the dataset
te = TwitterEmotion()
te.create_vocabulary(min_frequency=2)
w2v = te.w2v

te.train.open(fold=0)
te.validation.open(fold=0)
te.test.open(fold=0)

# Hyper Params

vocab_size = te.vocab_size
maxlen = 30
embedding_size = w2v.shape[-1]

# Convolution
kernel_size = 5
filters = 64
pool_size = 4

# LSTM
lstm_output_size = 70

# Training
batch_size = 500
epochs = 2

print('Building the Model...')
model = Sequential()
model.add(Embedding(vocab_size, embedding_size, input_length = maxlen,
					weights = [w2v]))
model.add(Dropout(0.25))
model.add(Conv1D(filters,
				 kernel_size,
				 padding = 'valid',
				 activation = 'relu',
				 strides = 1))
model.add(MaxPooling1D(pool_size = pool_size))
model.add(LSTM(lstm_output_size))
model.add(Dense(te.n_classes))
model.add(Activation('sigmoid'))

model.compile(loss = 'categorical_crossentropy',
			  optimizer = 'adam', metrics = ['accuracy'])

min_val_loss = float("inf")
prev_epoch = 0
while te.train.epochs_completed < epochs:

	train_batch = te.train.next_batch(batch_size = batch_size, pad = maxlen,
									  one_hot = True, mark_entities = True)
	[loss, accuracy] = model.train_on_batch(train_batch.text,
											train_batch.emotion)
	print('Epoch {}\tLoss: {}\tAcc: {}'.format(te.train.epochs_completed,
											   loss, accuracy))
	if prev_epoch != te.train.epochs_completed:
		prev_epoch = te.train.epochs_completed

		print('validating')
		total_val_loss, total_val_acc, n_val_iterations = 0.0, 0.0, 0
		while te.validation.epochs_completed < 1:
			val_batch = te.validation.next_batch(batch_size = batch_size,
												 pad = maxlen, one_hot = True,
												 mark_entities = True)
			[val_loss, val_accuracy] = model.test_on_batch(val_batch.text,
														   val_batch.emotion)

			total_val_loss += val_loss
			total_val_acc += val_accuracy
			n_val_iterations += 1
		te.validation._epochs_completed = 0
		avg_val_loss = total_val_loss / n_val_iterations
		avg_val_acc = total_val_acc / n_val_iterations
		print("Average Validation Loss: {}\t"
			  "Average Validation Accuracy: {}".format(avg_val_loss,
													   avg_val_acc))
		if avg_val_loss < min_val_loss:
			print('saving model as the validation loss improved. '
				  'Previous val loss: {}\t current val loss: {}'.format(
				min_val_loss, avg_val_loss))
			model.save('model_{}.h5'.format(te.train.epochs_completed))
			min_val_loss = avg_val_loss

print('Testing')
total_test_loss, total_test_acc, n_test_iterations = 0.0, 0.0, 0
while te.test.epochs_completed < 1:
	te.test._epochs_completed = 0
	test_batch = te.test.next_batch(batch_size = batch_size,
									pad = maxlen, one_hot = True,
									mark_entities = True)
	[test_loss, test_accuracy] = model.test_on_batch(test_batch.text,
													 test_batch.emotion)
	total_test_loss += test_loss
	total_test_acc += test_accuracy
	n_test_iterations += 1

avg_test_loss = total_test_loss / n_test_iterations
avg_test_acc = total_test_acc / n_test_iterations
print("Avg Test Accuracy: {}\nAverage Test Loss: {}".format(avg_test_acc,
															avg_test_loss))

te.train.close()
te.validation.close()
te.test.close()