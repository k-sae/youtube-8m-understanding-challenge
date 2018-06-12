from glob import glob
import tensorflow as tf

def parser(record, training=True):
    """
    In training mode labels will be returned, otherwise they won't be
    """
    keys_to_features = {
        "mean_rgb": tf.FixedLenFeature([1024], tf.float32),
        "mean_audio": tf.FixedLenFeature([128], tf.float32)
    }

    if training:
        keys_to_features["labels"] = tf.VarLenFeature(tf.int64)

    parsed = tf.parse_single_example(record, keys_to_features)
    x = tf.concat([parsed["mean_rgb"], parsed["mean_audio"]], axis=0)
    if training:
        y = tf.sparse_to_dense(parsed["labels"].values, [3862], 1)
        return x, y
    else:
        x = tf.concat([parsed["mean_rgb"], parsed["mean_audio"]], axis=0)
        return x


def make_datasetprovider(tf_records, repeats=1000, num_parallel_calls=12,
                         batch_size=32):
    """
    tf_records: list of strings - tf records you are going to use.
    repeats: how many times you want to iterate over the data.
    """
    dataset = tf.data.TFRecordDataset(tf_records)
    dataset = dataset.map(map_func=parser, num_parallel_calls=num_parallel_calls)
    dataset = dataset.repeat(repeats)

    dataset = dataset.shuffle(buffer_size=1000)
    dataset = dataset.batch(batch_size)

    d_iter = dataset.make_one_shot_iterator()
    return d_iter


def read_data(tf_records, batch_size=1, repeats=1000, num_parallel_calls=12, ):
    tf_records_glob = glob(tf_records)
    tf_provider = make_datasetprovider(tf_records_glob, repeats=repeats, num_parallel_calls=num_parallel_calls,
                                       batch_size=batch_size)
    sess = tf.Session()
    next_el = tf_provider.get_next()
    while True:
        try:
            yield sess.run(next_el)
        except tf.errors.OutOfRangeError:
            print("Iterations exhausted")
            break