from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tempfile

import pandas as pd
import tensorflow as tf

COLUMNS = ["col1", "col2", "col3", "col4", "col5", "col6", "col7", "col8", "col9", "col10",
           "col11", "col12", "col13", "col14", "col15", "col16", "col17", "col18", "col19", "col20",
           "col21", "col22", "col23", "col24", "col25", "col26", "col27", "col28", "col29", "col30",
           "col31", "col32", "col33", "col34", "col35", "col36", "col37", "col38", "col39", "col40"]
LABEL_COLUMN = "col1"
CATEGORICAL_COLUMNS = ["col15", "col16", "col17", "col18", "col19", "col20",
                       "col21", "col22", "col23", "col24", "col25", "col26", "col27", "col28", "col29", "col30",
                       "col31", "col32", "col33", "col34", "col35", "col36", "col37", "col38", "col39"]
CONTINUOUS_COLUMNS = ["col2", "col3", "col4", "col5", "col6", "col7", "col8", "col9", "col10",
                      "col11", "col12", "col13", "col14"]


def check_data_path(train_data, test_data):
    """Maybe downloads training data and returns train and test file names."""
    if train_data:
        train_file_name = train_data
    else:
        print("there is no training data path")

    if test_data:
        test_file_name = test_data
    else:
        print("there is no testing data path")

    return train_file_name, test_file_name


def build_estimator(model_dir, model_type):
    """Build an estimator."""

    # Continuous base columns.
    col2 = tf.contrib.layers.real_valued_column("col2")
    col3 = tf.contrib.layers.real_valued_column("col3")
    col4 = tf.contrib.layers.real_valued_column("col4")
    col5 = tf.contrib.layers.real_valued_column("col5")
    col6 = tf.contrib.layers.real_valued_column("col6")
    col7 = tf.contrib.layers.real_valued_column("col7")
    col8 = tf.contrib.layers.real_valued_column("col8")
    col9 = tf.contrib.layers.real_valued_column("col9")
    col10 = tf.contrib.layers.real_valued_column("col10")
    col11 = tf.contrib.layers.real_valued_column("col11")
    col12 = tf.contrib.layers.real_valued_column("col12")
    col13 = tf.contrib.layers.real_valued_column("col13")
    col14 = tf.contrib.layers.real_valued_column("col14")

    bucket_col16 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "col16", hash_bucket_size=1000)
    bucket_col17 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "col17", hash_bucket_size=1000)
    bucket_col18 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "col18", hash_bucket_size=1000)
    bucket_col19 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "col19", hash_bucket_size=1000)
    bucket_col20 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "col20", hash_bucket_size=1000)
    bucket_col21 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "col21", hash_bucket_size=1000)
    bucket_col22 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "col22", hash_bucket_size=1000)
    bucket_col23 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "col23", hash_bucket_size=1000)
    bucket_col24 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "col24", hash_bucket_size=1000)
    bucket_col25 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "col25", hash_bucket_size=1000)
    bucket_col26 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "col26", hash_bucket_size=1000)
    bucket_col27 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "col27", hash_bucket_size=1000)
    bucket_col28 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "col28", hash_bucket_size=1000)
    bucket_col29 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "col29", hash_bucket_size=1000)
    bucket_col30 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "col30", hash_bucket_size=1000)
    bucket_col31 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "col31", hash_bucket_size=1000)
    bucket_col32 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "col32", hash_bucket_size=1000)
    bucket_col33 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "col33", hash_bucket_size=1000)
    bucket_col34 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "col34", hash_bucket_size=1000)
    bucket_col35 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "col35", hash_bucket_size=1000)
    bucket_col36 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "col36", hash_bucket_size=1000)
    bucket_col37 = tf.contrib.layers.sparse_column_with_hash_bucket(
        "col37", hash_bucket_size=1000)

    # Wide columns and deep columns.
    wide_columns = [col2, col3, col4, col5, col6, col7, col8, col9, col10, col11, col12, col13, col14,
                    tf.contrib.layers.crossed_column([bucket_col30, bucket_col31],
                                                     hash_bucket_size=int(1e4)),
                    tf.contrib.layers.crossed_column(
                        [bucket_col30, bucket_col31, bucket_col32, bucket_col33],
                        hash_bucket_size=int(1e6))]
    deep_columns = [
        tf.contrib.layers.embedding_column(bucket_col16, dimension=8),
        tf.contrib.layers.embedding_column(bucket_col17, dimension=8),
        tf.contrib.layers.embedding_column(bucket_col18, dimension=8),
        tf.contrib.layers.embedding_column(bucket_col19, dimension=8),
        tf.contrib.layers.embedding_column(bucket_col20, dimension=8),
        tf.contrib.layers.embedding_column(bucket_col21, dimension=8),
        tf.contrib.layers.embedding_column(bucket_col22, dimension=8),
        tf.contrib.layers.embedding_column(bucket_col23, dimension=8),
        tf.contrib.layers.embedding_column(bucket_col24, dimension=8),
        tf.contrib.layers.embedding_column(bucket_col25, dimension=8),
        tf.contrib.layers.embedding_column(bucket_col26, dimension=8),
        tf.contrib.layers.embedding_column(bucket_col27, dimension=8),
        tf.contrib.layers.embedding_column(bucket_col28, dimension=8),
        tf.contrib.layers.embedding_column(bucket_col29, dimension=8),
        tf.contrib.layers.embedding_column(bucket_col30, dimension=8),
        tf.contrib.layers.embedding_column(bucket_col31, dimension=8),
        tf.contrib.layers.embedding_column(bucket_col32, dimension=8),
        tf.contrib.layers.embedding_column(bucket_col33, dimension=8),
        tf.contrib.layers.embedding_column(bucket_col34, dimension=8),
        tf.contrib.layers.embedding_column(bucket_col35, dimension=8),
        tf.contrib.layers.embedding_column(bucket_col36, dimension=8),
        tf.contrib.layers.embedding_column(bucket_col37, dimension=8),
    ]

    if model_type == "wide":
        m = tf.contrib.learn.LinearClassifier(model_dir=model_dir,
                                              feature_columns=wide_columns)
    elif model_type == "deep":
        m = tf.contrib.learn.DNNClassifier(model_dir=model_dir,
                                           feature_columns=deep_columns,
                                           hidden_units=[100, 50])
    else:
        m = tf.contrib.learn.DNNLinearCombinedClassifier(
            model_dir=model_dir,
            linear_feature_columns=wide_columns,
            dnn_feature_columns=deep_columns,
            dnn_hidden_units=[1000, 500, 100],
            fix_global_step_increment_bug=True)
    return m


def input_fn(df):
    """Input builder function."""
    # Creates a dictionary mapping from each continuous feature column name (k) to
    # the values of that column stored in a constant Tensor.
    continuous_cols = {k: tf.constant(df[k].values) for k in CONTINUOUS_COLUMNS}
    # Creates a dictionary mapping from each categorical feature column name (k)
    # to the values of that column stored in a tf.SparseTensor.
    categorical_cols = {
        k: tf.SparseTensor(
            indices=[[i, 0] for i in range(df[k].size)],
            values=df[k].values,
            dense_shape=[df[k].size, 1])
        for k in CATEGORICAL_COLUMNS}
    # Merges the two dictionaries into one.
    feature_cols = dict(continuous_cols)
    feature_cols.update(categorical_cols)
    # Converts the label column into a constant Tensor.
    label = tf.constant(df[LABEL_COLUMN].values)
    # Returns the feature columns and the label.
    return feature_cols, label


def train_and_eval(model_dir, model_type, train_steps, train_data, test_data):
    """Train and evaluate the model."""
    print("come into train_and_eval function")
    train_file_name, test_file_name = check_data_path(train_data, test_data)
    print("begin to read csv")
    df_train = pd.read_csv(
        tf.gfile.Open(train_file_name),
        names=COLUMNS,
        skipinitialspace=True,
        skiprows=1,
        engine="python")
    df_test = pd.read_csv(
        tf.gfile.Open(test_file_name),
        names=COLUMNS,
        skipinitialspace=True,
        skiprows=1,
        engine="python")

    # remove NaN elements
    df_train = df_train.dropna(how='any', axis=0)
    df_test = df_test.dropna(how='any', axis=0)
    print("begin to check:")

    #df_train[LABEL_COLUMN] = df_train["col1"].astype(int)
    #print("df_train checked")
    #df_test[LABEL_COLUMN] = df_test["col1"].astype(int)
    #print("df_test checked")

    model_dir = tempfile.mkdtemp() if not model_dir else model_dir
    print("model directory = %s" % model_dir)

    m = build_estimator(model_dir, model_type)
    m.fit(input_fn=lambda: input_fn(df_train), steps=train_steps)
    results = m.evaluate(input_fn=lambda: input_fn(df_test), steps=1)
    for key in sorted(results):
        print("%s: %s" % (key, results[key]))


FLAGS = None


def main(_):
    train_and_eval(FLAGS.model_dir, FLAGS.model_type, FLAGS.train_steps,
                   FLAGS.train_data, FLAGS.test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="",
        help="Base directory for output models."
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="wide_n_deep",
        help="Valid model types: {'wide', 'deep', 'wide_n_deep'}."
    )
    parser.add_argument(
        "--train_steps",
        type=int,
        default=200,
        help="Number of training steps."
    )
    parser.add_argument(
        "--train_data",
        type=str,
        default="",
        help="Path to the training data."
    )
    parser.add_argument(
        "--test_data",
        type=str,
        default="",
        help="Path to the test data."
    )
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
