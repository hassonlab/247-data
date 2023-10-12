import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import pickle
from typing import Sequence

import pandas as pd
import tensorflow as tf
from absl import app, flags, logging

## Globals
DATA_DIR = "/projects/HASSON/247/data/conversations-car/"

EXCLUDE_WORDS = ["sp", "{lg}", "{ns}", "{LG}", "{NS}", "SP"]
NON_WORDS = ["hm", "huh", "mhm", "mm", "oh", "uh", "uhuh", "um"]

## Flags (command-line arguments)
FLAGS = flags.FLAGS
flags.DEFINE_string("patient_id", "", "Patient ID")
flags.DEFINE_enum(
    "input_format",
    "pickles",
    ["raw", "pickles", "edf"],
    "Format specifier for reading inputs",
)


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):  # if value is tensor
        value = value.numpy()  # get value of tensor
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a floast_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _int64_feature(value):
    """Returns an int64_list from a bool / enum / int / uint."""
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _string_feature(value):
    """Returns a byte_list from a string."""
    value = value.encode("utf-8")
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example(signal, word, spkr, start, end, emb=None):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the
    # tf.train.Example-compatible data type.
    feature = {
        # "emb": _float_feature(emb),
        "signal": _float_feature(signal),
        "onset": _float_feature(start),  ## FIXME onset cannot be float
        "offset": _float_feature(end),  ## FIXME offset cannot be float
        "word": _string_feature(word),
        "speaker": _string_feature(spkr),
    }
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )
    return example_proto.SerializeToString()


def load_pickle(pickle_name, key=None):
    """Load the datum pickle and returns as a dataframe

    Args:
        file (string): labels pickle from 247-decoding/tfs_pickling.py

    Returns:
        DataFrame: pickle contents returned as dataframe
    """
    with open(pickle_name, "rb") as fh:
        datum = pickle.load(fh)

    if key:
        df = pd.DataFrame.from_dict(datum[key])
    else:
        df = pd.DataFrame.from_dict(datum)

    return df


def load_raw():
    ## If reading for raw stimuli and response
    logging.info("Reading from mat files")

    CONV_DIRS = os.path.join(DATA_DIR, FLAGS.patient_id)
    OUT_DIR = os.path.join(os.getcwd(), "results", FLAGS.patient_id)

    logging.info("Creating Output Directory")
    os.makedirs(OUT_DIR, exist_ok=True)

    logging.info(CONV_DIRS)


def load_pickles() -> list:
    ## If reading from already created pickles
    logging.info("Reading from pickles files")

    datum_pickle = "/scratch/gpfs/hgazula/247-pickling/results/tfs/625/pickles/embeddings/gpt2-xl/full/base_df.pkl"

    df = load_pickle(datum_pickle)
    df_dict = df.to_dict(orient="records")

    keys = ["word", "speaker", "onset", "offset"]

    serialized_examples = []
    for item in df_dict:
        serialized_examples.append(
            serialize_example(*([0] + [item.get(key) for key in keys]))
        )

    return serialized_examples


def load_edf():
    ## if reading from EDF and transcript
    logging.info("Reading from edf files")
    pass


def write_to_tfr(examples: list) -> None:
    filename = "247-625.tfrecords"  ## FIXME: Find a better name for the files

    with tf.io.TFRecordWriter(filename) as writer:
        for example in examples:
            writer.write(example)
    writer.close()

    logging.info(
        f"Wrote {len(examples)} elements to {os.path.join(os.getcwd(), filename)}"
    )


def main(argv: Sequence) -> None:
    del argv

    input_format_dict = {
        "raw": load_raw,
        "pickles": load_pickles,
        "edf": load_edf,
    }

    file_name = f""

    serialized_examples = input_format_dict[FLAGS.input_format]()
    write_to_tfr(serialized_examples)


if __name__ == "__main__":
    # flags.mark_flag_as_required("patient_id")
    app.run(main)
