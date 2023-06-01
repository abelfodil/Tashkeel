def download_file_if_not_exists(url, output):
    import os.path

    if os.path.isfile(output):
        print(f"{output} already exists. Skipping download.")
        return
    
    import urllib.request

    urllib.request.urlretrieve(url, output)


def convert_pickle_to_json(input_pickle_path, output_json_path):
    import json
    import pickle
    
    with open(input_pickle_path, 'rb') as pickle_file:
        unpickled = pickle.load(pickle_file)

    with open(output_json_path, 'w', encoding='utf8') as json_file:
        json.dump(unpickled, json_file, ensure_ascii=False)


def convert_keras_tf1_model_to_onnx(input_model_path, output_model_path):
    from tensorflow.compat.v1.keras.models import load_model
    import tf2onnx
    
    model = load_model(input_model_path)
    tf2onnx.convert.from_keras(model, output_path=output_model_path)


download_file_if_not_exists('https://raw.githubusercontent.com/Barqawiz/Shakkala/master/shakkala/dictionary/input_vocab_to_int.pickle', 'raw_model/input_map.pickle')
convert_pickle_to_json('raw_model/input_map.pickle', 'output_model/input_map.json')

download_file_if_not_exists('https://raw.githubusercontent.com/Barqawiz/Shakkala/master/shakkala/dictionary/output_int_to_vocab.pickle', 'raw_model/output_map.pickle')
convert_pickle_to_json('raw_model/output_map.pickle', 'output_model/output_map.json')

download_file_if_not_exists('https://raw.githubusercontent.com/Barqawiz/Shakkala/master/shakkala/model/second_model6.h5', 'raw_model/model.h5')
convert_keras_tf1_model_to_onnx('raw_model/model.h5', 'output_model/model.onnx')
