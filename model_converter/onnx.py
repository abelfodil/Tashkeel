import onnxruntime as ort
import onnx
import json

input_text = "اللُّغَةُ العَرَبِيَّة هي أكثر اللغات السامية تحدثًا، وإحدى أكثر اللغات انتشاراً في العالم، يتحدثها أكثر من 467 مليون نسمة"

model = ort.InferenceSession('output_model/model.onnx')


with open('output_model/input_map.json', encoding='utf-8') as fh:
    input_vocab_to_int = json.load(fh)
    


logits = model.run(None, {'embedding_7_input':input_int})[0][0]

