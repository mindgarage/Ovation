# This is a modified version of the `vocabulary_expansion.py` file python file
# found in
# https://github.com/tensorflow/models/blob/master/skip_thoughts/skip_thoughts/vocabulary_expansion.py
#
# The following are the changes performed in the file:
# (this summmary is required by the Apache License)
#
#  * Removes most of the references to the `skip_thoughts` name.
#  * Adds variable `embedding_tensor_name`.
#  * Adds code to print the tensor names
#  * Uses `embedding_tensor_name` in load_model_embeddings
#

# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


"""
This script loads the word embeddings from a trained model and
from a trained word2vec model (typically with a larger vocabulary). It trains a
linear regression model without regularization to learn a linear mapping from
the word2vec embedding space to the model's embedding space. The model is
then applied to all words in the word2vec vocabulary, yielding vectors in the
model word embedding space for the union of the two vocabularies.

The linear regression task is to learn a parameter matrix W to minimize
  || X - Y * W ||^2,
where X is a matrix of the models embeddings of shape [num_words, dim1],
Y is a matrix of word2vec embeddings of shape [num_words, dim2], and W is a
matrix of shape [dim2, dim1].

This is based on the "Translation Matrix" method from the paper:

  "Exploiting Similarities among Languages for Machine Translation"
  Tomas Mikolov, Quoc V. Le, Ilya Sutskever
  https://arxiv.org/abs/1309.4168
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os.path


import gensim.models
import numpy as np
import sklearn.linear_model
import tensorflow as tf

from tensorflow.python import pywrap_tensorflow

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("model_checkpoint", None,
                       "Checkpoint file or directory containing a checkpoint "
                       "file.")
tf.flags.DEFINE_string("embedding_tensor_name", 'W_embedding',
                       "Checkpoint file or directory containing a checkpoint "
                       "file.")
tf.flags.DEFINE_string("vocab", None,
                       "Path to vocabulary file containing a list of newline-"
                       "separated words where the word id is the "
                       "corresponding 0-based index in the file.")

tf.flags.DEFINE_string("word2vec_model", None,
                       "File containing a word2vec model in binary format.")

tf.flags.DEFINE_string("output_dir", None, "Output directory.")

tf.logging.set_verbosity(tf.logging.INFO)


def _load_model_embeddings(checkpoint_path, embedding_tensor_name):
  """Loads the embedding matrix from a model checkpoint.

  Args:
    checkpoint_path: Model checkpoint file or directory containing a checkpoint
        file.

  Returns:
    word_embedding: A numpy array of shape [vocab_size, embedding_dim].

  Raises:
    ValueError: If no checkpoint file matches checkpoint_path.
  """
  if tf.gfile.IsDirectory(checkpoint_path):
    checkpoint_file = tf.train.latest_checkpoint(checkpoint_path)
    if not checkpoint_file:
      raise ValueError("No checkpoint file found in %s" % checkpoint_path)
  else:
    checkpoint_file = checkpoint_path

  tf.logging.info("Loading the model's embedding matrix from %s",
                  checkpoint_file)
  reader = tf.train.NewCheckpointReader(checkpoint_file)
  var_to_shape_map = reader.get_variable_to_shape_map()
  for key in var_to_shape_map :
    print("tensor_name: ", key)
    print(reader.get_tensor(key).shape)
  word_embedding = reader.get_tensor("{}".format(embedding_tensor_name))
  tf.logging.info("Loaded model embedding matrix of shape %s",
                  word_embedding.shape)

  return word_embedding


def _load_vocabulary(filename):
  """Loads a vocabulary file.

  Args:
    filename: Path to text file containing newline-separated words.

  Returns:
    vocab: A dictionary mapping word to word id.
  """
  tf.logging.info("Reading vocabulary from %s", filename)
  vocab = collections.OrderedDict()
  with tf.gfile.GFile(filename, mode="r") as f:
    for i, line in enumerate(f):
      word = line.strip().split('\t')[0]
      assert word not in vocab, "Attempting to add word twice: %s" % word
      vocab[word] = i
  tf.logging.info("Read vocabulary of size %d", len(vocab))
  return vocab


def _expand_vocabulary(model_emb, models_vocab, word2vec):
  """Runs vocabulary expansion on a model using a word2vec model.

  Args:
    model_emb: A numpy array of shape [models_vocab_size,
        model_embedding_dim].
    models_vocab: A dictionary of word to id.
    word2vec: An instance of gensim.models.Word2Vec.

  Returns:
    combined_emb: A dictionary mapping words to embedding vectors.
  """
  # Find words shared between the two vocabularies.
  tf.logging.info("Finding shared words")
  shared_words = [w for w in word2vec.vocab if w in models_vocab]

  # Select embedding vectors for shared words.
  tf.logging.info("Selecting embeddings for %d shared words", len(shared_words))
  shared_st_emb = model_emb[[
      models_vocab[w] for w in shared_words
  ]]
  shared_w2v_emb = word2vec[shared_words]

  # Train a linear regression model on the shared embedding vectors.
  tf.logging.info("Training linear regression model")
  model = sklearn.linear_model.LinearRegression()
  model.fit(shared_w2v_emb, shared_st_emb)

  # Create the expanded vocabulary.
  tf.logging.info("Creating embeddings for expanded vocabuary")
  combined_emb = collections.OrderedDict()
  for w in word2vec.vocab:
    # Ignore words with underscores (spaces).
    if "_" not in w:
      w_emb = model.predict(word2vec[w].reshape(1, -1))
      combined_emb[w] = w_emb.reshape(-1)

  for w in models_vocab:
    combined_emb[w] = model_emb[models_vocab[w]]

  tf.logging.info("Created expanded vocabulary of %d words", len(combined_emb))

  return combined_emb


def main(unused_argv):
  if not FLAGS.model_checkpoint:
    raise ValueError("--model_checkpoint is required.")
  if not FLAGS.vocab:
    raise ValueError("--vocab is required.")
  if not FLAGS.embedding_tensor_name:
    raise ValueError("--embedding_tensor_name is required.")
  if not FLAGS.word2vec_model:
    raise ValueError("--word2vec_model is required.")
  if not FLAGS.output_dir:
    raise ValueError("--output_dir is required.")

  if not tf.gfile.IsDirectory(FLAGS.output_dir):
    tf.gfile.MakeDirs(FLAGS.output_dir)

  # Load the model embeddings and vocabulary.
  model_emb = _load_model_embeddings(FLAGS.model_checkpoint,
                                     FLAGS.embedding_tensor_name)
  vocab = _load_vocabulary(FLAGS.vocab)

  # Load the Word2Vec model.
  word2vec = gensim.models.KeyedVectors.load_word2vec_format(
      FLAGS.word2vec_model, binary=True)

  # Run vocabulary expansion.
  embedding_map = _expand_vocabulary(model_emb, vocab,
                                     word2vec)

  # Save the output.
  vocab = embedding_map.keys()
  vocab_file = os.path.join(FLAGS.output_dir, "vocab.txt")
  with tf.gfile.GFile(vocab_file, "w") as f:
    f.write("\n".join(vocab))
  tf.logging.info("Wrote vocabulary file to %s", vocab_file)

  embeddings = np.array(embedding_map.values())
  embeddings_file = os.path.join(FLAGS.output_dir, "embeddings.npy")
  np.save(embeddings_file, embeddings)
  tf.logging.info("Wrote embeddings file to %s", embeddings_file)


if __name__ == "__main__":
  tf.app.run()
