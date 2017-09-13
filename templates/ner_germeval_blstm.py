import os
import datetime
import tflearn

import tensorflow as tf

from datasets import Germeval
from datasets import id2seq
from models import BLSTMGermEval
from datasets import onehot2seq


# Model Parameters
tf.flags.DEFINE_integer("embedding_dim", 300, "Dimensionality of character "
                                            "embedding (default: 300)")
tf.flags.DEFINE_boolean("train_embeddings", True, "True if you want to train "
                                                  "the embeddings False "
                                                  "otherwise")
tf.flags.DEFINE_float("dropout", 0.5, "Dropout keep probability ("
                                              "default: 1.0)")
tf.flags.DEFINE_float("l2_reg_beta", 0.0, "L2 regularizaion lambda ("
                                            "default: 0.0)")
tf.flags.DEFINE_integer("hidden_units", 128, "Number of hidden units of the "
                                             "RNN Cell")
tf.flags.DEFINE_integer("rnn_layers", 2, "Number of layers in the RNN")
tf.flags.DEFINE_string("optimizer", 'adam', "Which Optimizer to use. "
                    "Available options are: adam, gradient_descent, adagrad, "
                    "adadelta, rmsprop")
tf.flags.DEFINE_integer("learning_rate", 0.0001, "Learning Rate")
tf.flags.DEFINE_integer("sequence_length", 50, "maximum length of a sequence")

# Training parameters
tf.flags.DEFINE_integer("max_checkpoints", 100, "Maximum number of "
                                                "checkpoints to save.")
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("num_epochs", 300, "Number of training epochs"
                                           " (default: 200)")
tf.flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set "
                                    "after this many steps (default: 100)")
tf.flags.DEFINE_integer("checkpoint_every", 100, "Save model after this many"
                                                  " steps (default: 100)")
tf.flags.DEFINE_integer("max_dev_itr", 100, "max munber of dev iterations "
                              "to take for in-training evaluation")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft"
                                                      " device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops"
                                                       " on devices")
tf.flags.DEFINE_boolean("verbose", True, "Log Verbosity Flag")
tf.flags.DEFINE_float("gpu_fraction", 0.5, "Fraction of GPU to use")
tf.flags.DEFINE_string("data_dir", "/scratch", "path to the root of the data "
                                           "directory")
tf.flags.DEFINE_string("experiment_name", "NER_GERMEVAL_BLSTM",
                       "Name of your model")
tf.flags.DEFINE_string("mode", "train", "'train' or 'test'")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()


def initialize_tf_graph(metadata_path, w2v, n_classes):
    config = tf.ConfigProto(
        allow_soft_placement=FLAGS.allow_soft_placement,
        log_device_placement=FLAGS.log_device_placement)
    config.gpu_options.per_process_gpu_memory_fraction = FLAGS.gpu_fraction
    sess = tf.Session(config=config)
    print("Session Started")

    with sess.as_default():
        args = FLAGS.__flags
        args['n_classes'] = n_classes
        ner_model = BLSTMGermEval(args)
        ner_model.show_train_params()
        ner_model.build_model(metadata_path=metadata_path,
                              embedding_weights=w2v)
        ner_model.create_optimizer()
        print("Siamese CNN LSTM Model built")

    print('Setting Up the Model. You can do it one at a time. In that case '
          'drill down this method')
    ner_model.easy_setup(sess)
    return sess, ner_model


def train(dataset, metadata_path, w2v, n_classes):
    print("Configuring Tensorflow Graph")
    with tf.Graph().as_default():

        sess, ner_model = initialize_tf_graph(metadata_path, w2v, n_classes)

        min_validation_loss = float("inf")
        avg_val_loss = 0.0
        prev_epoch = 0
        tflearn.is_training(True, session=sess)
        while dataset.train.epochs_completed < FLAGS.num_epochs:
            train_batch = dataset.train.next_batch(batch_size=FLAGS.batch_size,
                        pad=ner_model.args["sequence_length"], one_hot=True)
            pred, loss, step, acc = ner_model.train_step(sess,
                                train_batch.sentences, train_batch.ner1,
                                    train_batch.lengths, dataset.train.epochs_completed)

            if step % FLAGS.evaluate_every == 0:
                avg_val_loss, avg_val_acc, _ = evaluate(sess=sess,
                             dataset=dataset.validation, model=ner_model,
                                max_dev_itr=FLAGS.max_dev_itr, mode='val',
                                    step=step)

            if step % FLAGS.checkpoint_every == 0:
                min_validation_loss = maybe_save_checkpoint(sess,
                    min_validation_loss, avg_val_loss, step, ner_model)

            if dataset.train.epochs_completed != prev_epoch:
                prev_epoch = dataset.train.epochs_completed
                avg_test_loss, avg_test_acc, _ = evaluate(
                            sess=sess, dataset=dataset.test, model=ner_model,
                            max_dev_itr=0, mode='test', step=step)
                min_validation_loss = maybe_save_checkpoint(sess,
                            min_validation_loss, avg_val_loss, step, ner_model)

def maybe_save_checkpoint(sess, min_validation_loss, val_loss, step, model):
    if val_loss <= min_validation_loss:
        model.saver.save(sess, model.checkpoint_prefix, global_step=step)
        tf.train.write_graph(sess.graph.as_graph_def(), model.checkpoint_prefix,
                             "graph" + str(step) + ".pb", as_text=False)
        print("Saved model {} with avg_loss={} checkpoint"
              " to {}\n".format(step, min_validation_loss,
                                model.checkpoint_prefix))
        return val_loss
    return min_validation_loss


def evaluate(sess, dataset, model, step, max_dev_itr=100, verbose=True,
             mode='val'):
    results_dir = model.val_results_dir if mode == 'val' \
        else model.test_results_dir
    samples_path = os.path.join(results_dir,
                                '{}_samples_{}.txt'.format(mode, step))
    history_path = os.path.join(results_dir,
                                '{}_history.txt'.format(mode))

    avg_val_loss, avg_acc = 0.0, 0.0
    print("Running Evaluation {}:".format(mode))
    tflearn.is_training(False, session=sess)

    # This is needed to reset the local variables initialized by
    # TF for calculating streaming Pearson Correlation and MSE
    all_dev_text, all_dev_pred, all_dev_gt = [], [], []
    dev_itr = 0
    while (dev_itr < max_dev_itr and max_dev_itr != 0) \
            or mode in ['test', 'train']:
        val_batch = dataset.next_batch(FLAGS.batch_size,
                                       pad=model.args["sequence_length"],
                                       one_hot=True, raw=False)
        loss, pred, acc = model.evaluate_step(sess, val_batch.sentences,
                                              val_batch.ner1, val_batch.lengths)
        avg_val_loss += loss
        avg_acc += acc
        all_dev_text += id2seq(val_batch.sentences, dataset.vocab_i2w[0])
        all_dev_pred += onehot2seq(pred, dataset.vocab_i2w[2])
        all_dev_gt += onehot2seq(val_batch.ner1, dataset.vocab_i2w[2])
        dev_itr += 1

        if mode == 'test' and dataset.epochs_completed == 1: break
        if mode == 'train' and dataset.epochs_completed == 1: break

    result_set = (all_dev_text, all_dev_pred, all_dev_gt)
    avg_loss = avg_val_loss / dev_itr
    avg_acc = avg_acc / dev_itr
    if verbose:
        print("{}:\t Loss: {}".format(mode, avg_loss, avg_acc))

    with open(samples_path, 'w') as sf, open(history_path, 'a') as hf:
        for x1, pred, gt in zip(all_dev_text, all_dev_pred, all_dev_gt):
            sf.write('{}\t{}\t{}\n'.format(x1, pred, gt))
        hf.write('STEP:{}\tTIME:{}\tacc:{}\tLoss\t{}\n'.format(
                step, datetime.datetime.now().isoformat(),
                avg_acc, avg_loss))
    tflearn.is_training(True, session=sess)
    return avg_loss, avg_acc, result_set


def test(dataset, metadata_path, w2v, n_classes):
    print("Configuring Tensorflow Graph")
    with tf.Graph().as_default():
        sess, siamese_model = initialize_tf_graph(metadata_path, w2v, n_classes)
        avg_test_loss, avg_test_acc, test_result_set = evaluate(sess=sess,
                                    dataset=dataset.test, model=siamese_model,
                                        max_dev_itr=0, mode='test', step=-1)
        print('Average acc score: {}\nAverage Loss: {}'.format(
                avg_test_acc, avg_test_loss))

if __name__ == '__main__':
    germeval = Germeval()

    if FLAGS.mode == 'train':
        train(germeval, germeval.metadata_paths, germeval.w2v,
                          len(germeval.w2i[1]))
    elif FLAGS.mode == 'test':
        test(germeval, germeval.metadata_paths, germeval.w2v,
                         len(germeval.w2i[1]))
    else:
        raise ValueError('Mode {} is not defined'.format(FLAGS.mode))