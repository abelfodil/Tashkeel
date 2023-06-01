import os
import tensorflow as tf
import numpy as np
import json
from tensorflow.compat.v1.keras.models import load_model

def combine_text_with_harakat(input_sent, output_sent):
    text = ""
    for character, haraka in zip(input_sent, output_sent):
        if haraka == '<UNK>' or haraka == 'ـ':
            haraka = ''
        text += character + "" + haraka

    return text

def build_preprocessor():
    with open('output_model/input_map.json', encoding='utf-8') as fh:
        input_vocab_to_int = json.load(fh)

    vocab = [k for k, _ in sorted(((k, v) for k, v in input_vocab_to_int.items()), key=lambda x: x[1])][1:]

    return tf.keras.layers.TextVectorization(
        standardize=None,
        split='character',
        output_mode='int',
        output_sequence_length=315,
        vocabulary=vocab
    )

def build_postprocessor():
    with open('output_model/output_map.json', encoding='utf-8') as fh:
        output_int_to_vocab = json.load(fh)

    vocab = [v for _, v in sorted(((k, v) for k, v in output_int_to_vocab.items()), key=lambda x: int(x[0]))]

    return tf.keras.layers.StringLookup(
        vocabulary=vocab,
        num_oov_indices=0,
        invert=True,
    )

def logits_to_text(logits):
    text = []
    for sentence in logits.numpy():
        for prediction in sentence:
            if prediction == b'<PAD>':
                continue
            text.append(prediction.decode("utf-8"))
    return text

def get_final_text(input_sent, output_sent):
    return combine_text_with_harakat(input_sent, output_sent)

import numpy as np

if __name__ == "__main__":
    input_text = "اللُّغَةُ العَرَبِيَّة هي أكثر اللغات السامية تحدثًا، وإحدى أكثر اللغات انتشاراً في العالم، يتحدثها أكثر من 467 مليون نسمة"

    print("finished preparing input")

    print("start with model")

    model = load_model('raw_model/model.h5')

    new_model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(1,), dtype=tf.string),
            build_preprocessor(),
            model,
            tf.keras.layers.Lambda(lambda x: tf.argmax(x, -1)),
            build_postprocessor(),
        ]
    )

    # import tf2onnx
    # tf2onnx.convert.from_keras(new_model, output_path='output_model/model.onnx')

    # logits = model.run(None, {'embedding_7_input':input_int})[0][0]
    logits = new_model(tf.constant([input_text]))

    print("prepare and print output")
    predicted_harakat = logits_to_text(logits)

    final_output = get_final_text(input_text, predicted_harakat)
    with open('output.txt', 'wt') as out:
        out.write(final_output)
    print(final_output)

    print("finished successfully")
