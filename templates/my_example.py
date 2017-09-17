import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Embedding
from keras.layers import LSTM
from datasets import TwitterEmotion

te = TwitterEmotion()

te.train.open()
te.validation.open()
te.test.open()

model = Sequential()
model.add(Embedding(te.vocab_size, 128, input_length=25))
model.add(LSTM(128))
model.add(Dense(te.n_classes, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])


while te.train.epochs_completed < 10:
    train_batch = te.train.next_batch(one_hot=True, pad=25)
    [loss, acc] = model.train_on_batch(train_batch.text,
                                       train_batch.emotion)

    print('TRAIN LOSS: {}\t TRAIN ACC: {}'.format(loss, acc))


test_losses, test_acc = [], []
while te.test.epochs_completed < 1:
    test_batch = te.test.next_batch(one_hot=True, pad=25)
    [loss, acc] = model.train_on_batch(test_batch.text,
                                       test_batch.emotion)

    print('TEST LOSS: {}\t TEST ACC: {}'.format(loss, acc))
    test_losses.append(loss)
    test_acc.append(acc)

print('Mean Avg Tests: {}\t Mean Avg Acc: {}'.format(np.mean(test_losses),
                                                     np.mean(test_acc)))


