import pathlib
import pickle
import tempfile
from collections import namedtuple
from datetime import datetime

import apache_beam as beam
import sklearn
import sklearn.random_projection as pr
import tensorflow
import tensorflow_hub as hub
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
from apache_beam.options.pipeline_options import PipelineOptions

encoder = None
module_url = 'https://tfhub.dev/google/universal-sentence-encoder/2'  # @param {type:"string"}
projected_dim = 64  # @param {type:"number"}


def load_module(module_url):
    embed_module = hub.Module(module_url)
    placeholder = tensorflow.compat.v1.placeholder(dtype=tensorflow.compat.v1.string)
    embed = embed_module(placeholder)
    session = tensorflow.compat.v1.Session()
    session.run([tensorflow.compat.v1.global_variables_initializer(), tensorflow.compat.v1.tables_initializer()])
    print('TF-Hub module is loaded.')

    def _embeddings_fn(sentences):
        computed_embeddings = session.run(
            embed, feed_dict={placeholder: sentences})
        return computed_embeddings

    return _embeddings_fn


def generate_random_projection_weights(original_dim, projected_dim):
    random_projection_matrix = None
    if projected_dim and original_dim > projected_dim:
        random_projection_matrix = pr._gaussian_random_matrix(projected_dim, original_dim).T
        print("A Gaussian random weight matrix was creates with shape of {}".format(random_projection_matrix.shape))
        print('Storing random projection matrix to disk...')
        with open('random_projection_matrix', 'wb') as handle:
            pickle.dump(random_projection_matrix,
                        handle, protocol=pickle.HIGHEST_PROTOCOL)

    return random_projection_matrix


def create_metadata():
    '''Creates metadata for the raw data'''
    from tensorflow_transform.tf_metadata import dataset_metadata
    from tensorflow_transform.tf_metadata import schema_utils
    feature_spec = {'text': tensorflow.compat.v1.FixedLenFeature([], dtype=tensorflow.compat.v1.string)}
    schema = schema_utils.schema_from_feature_spec(feature_spec)
    metadata = dataset_metadata.DatasetMetadata(schema)
    return metadata


def embed_text(text, module_url, random_projection_matrix):
    # Beam will run this function in different processes that need to
    # import hub and load embed_fn (if not previously loaded)
    global encoder
    if not encoder:
        encoder = hub.Module(module_url)
    embedding = encoder(text)
    if random_projection_matrix is not None:
        # Perform random projection for the embedding
        embedding = tensorflow.compat.v1.matmul(
            embedding, tensorflow.compat.v1.cast(random_projection_matrix, embedding.dtype))
    return embedding


def make_preprocess_fn(module_url, random_projection_matrix=None):
    '''Makes a tft preprocess_fn'''

    def _preprocess_fn(input_features):
        '''tft preprocess_fn'''
        text = input_features['text']
        # Generate the embedding for the input text
        embedding = embed_text(text, module_url, random_projection_matrix)
        output_features = {
            'text': text,
            'embedding': embedding
        }
        return output_features

    return _preprocess_fn


def run_hub2emb(args):
    '''Runs the embedding generation pipeline'''

    options = beam.options.pipeline_options.PipelineOptions(**args)
    args = namedtuple("options", args.keys())(*args.values())

    raw_metadata = create_metadata()
    converter = tft.coders.CsvCoder(
        column_names=['text'], schema=raw_metadata.schema)

    with beam.Pipeline(args.runner, options=options) as pipeline:
        with tft_beam.Context(args.temporary_dir):
            # Read the sentences from the input file
            sentences = (
                    pipeline
                    | 'Read sentences from files' >> beam.io.ReadFromText(
                file_pattern='corpus/text.txt')
                   # | 'Convert to dictionary' >> beam.Map(converter.decode)
            )

            sentences_dataset = (sentences, raw_metadata)
            preprocess_fn = make_preprocess_fn(args.module_url, args.random_projection_matrix)
            # Generate the embeddings for the sentence using the TF-Hub module
            embeddings_dataset, _ = (
                    sentences_dataset
                    | 'Extract embeddings' >> tft_beam.AnalyzeAndTransformDataset(preprocess_fn)
            )

            embeddings, transformed_metadata = embeddings_dataset
            # Write the embeddings to TFRecords files
            embeddings | 'Write embeddings to TFRecords' >> beam.io.tfrecordio.WriteToTFRecord(
                file_path_prefix='{}/emb'.format(args.output_dir),
                file_name_suffix='.tfrecords',
                coder=tft.coders.ExampleProtoCoder(transformed_metadata.schema))


# ======================================================================
output_dir = pathlib.Path(tempfile.mkdtemp())
temporary_dir = pathlib.Path(tempfile.mkdtemp())
g = tensorflow.compat.v1.Graph()
with g.as_default():
    original_dim = load_module(module_url)(['']).shape[1]
    random_projection_matrix = None

    if projected_dim:
        random_projection_matrix = generate_random_projection_weights(
            original_dim, projected_dim)

args = {
    'job_name': 'hub2emb-{}'.format(datetime.utcnow().strftime('%y%m%d-%H%M%S')),
    'runner': 'DirectRunner',
    'batch_size': 1024,
    'data_dir': 'corpus/*.txt',
    'output_dir': output_dir,
    'temporary_dir': temporary_dir,
    'module_url': module_url,
    'random_projection_matrix': random_projection_matrix,
}

print("Pipeline args are set.")
print(args)
import os
print(os.getcwd())
print("Running pipeline...")
run_hub2emb(args)
print("Pipeline is done.")
