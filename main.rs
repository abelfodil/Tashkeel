use ndarray::{try_into, Array, ArrayBase, Axis, Ix2, Ix3, OwnedRepr};
use ort::{
    tensor::{DynOrtTensor, FromArray, InputTensor, OrtOwnedTensor},
    Environment, ExecutionProvider, GraphOptimizationLevel, OrtResult, SessionBuilder,
};
use serde_json::Value;
use std::collections::HashMap;
use std::fs;
use wasm_bindgen::prelude::*;

fn split_sentences(input: &str) -> Vec<Vec<char>> {
    input
        .split('.')
        .filter(|s| !s.is_empty())
        .map(|sentence| sentence.chars().collect())
        .collect()
}

fn merge_characters(input_sentences: Vec<Vec<char>>, output_sentences: Vec<Vec<char>>) -> String {
    input_sentences
        .into_iter()
        .zip(output_sentences.into_iter())
        .map(|(input_sentence, output_sentence)| {
            input_sentence
                .into_iter()
                .zip(output_sentence.into_iter())
                .map(|(input_char, output_char)| {
                    // convert characters to their Unicode code points, add them and convert back to a character
                    let sum = (input_char as u32) + (output_char as u32);
                    std::char::from_u32(sum).unwrap_or('\0')
                })
                .collect::<String>()
        })
        .collect::<Vec<String>>()
        .join(". ")
}

// Struct for vectorizing the input
pub struct InputVectorizer {
    pub char_map: HashMap<char, i32>,
}

impl InputVectorizer {
    pub fn new(json_path: &str) -> Result<Self, JsValue> {
        let file = fs::read_to_string(json_path)
            .map_err(|_| JsValue::from_str("Failed to read JSON file"))?;
        let v: Value = serde_json::from_str(&file)
            .map_err(|_| JsValue::from_str("Failed to parse JSON file"))?;
        let char_map: HashMap<char, i32> = serde_json::from_value(v)
            .map_err(|_| JsValue::from_str("Failed to create character map"))?;
        Ok(Self { char_map })
    }

    pub fn process(&self, input_chars: Vec<Vec<char>>) -> Result<Vec<Vec<i32>>, JsValue> {
        let input_embeddings = input_chars
            .into_iter()
            .map(|sentence_chars| {
                let mut sentence_embeddings: Vec<i32> = sentence_chars
                    .into_iter()
                    .map(|c| self.char_map.get(&c).cloned().unwrap_or(1))
                    .collect();

                while sentence_embeddings.len() < 315 {
                    sentence_embeddings.push(0);
                }

                Ok(sentence_embeddings)
            })
            .collect::<Result<Vec<Vec<i32>>>, JsValue>()?;

        Ok(input_embeddings)
    }
}

// Struct for mapping output to strings
pub struct StringMapper {
    pub string_map: Vec<char>,
}

impl StringMapper {
    pub fn new(json_path: &str) -> Result<Self, JsValue> {
        let file = fs::read_to_string(json_path)
            .map_err(|_| JsValue::from_str("Failed to read JSON file"))?;
        let v: Value = serde_json::from_str(&file)
            .map_err(|_| JsValue::from_str("Failed to parse JSON file"))?;
        let string_map: HashMap<i32, char> = serde_json::from_value(v)
            .map_err(|_| JsValue::from_str("Failed to create string map"))?;

        let mut string_vec: Vec<(i32, char)> = string_map.into_iter().collect();
        string_vec.sort_by_key(|&(key, _)| key);

        let string_map: Vec<char> = string_vec.into_iter().map(|(_, value)| value).collect();

        Ok(Self { string_map })
    }

    pub fn process(&self, logits: ArrayBase<OwnedRepr<f32>, Ix3>) -> Vec<Vec<char>> {
        let argmax = argmax_axis(&logits, Axis(2));
        argmax
            .into_iter()
            .map(|row| {
                row.into_iter()
                    .map(|index| self.string_map[index as usize])
                    .collect()
            })
            .collect()
    }
}

fn argmax_axis<T: PartialOrd>(
    array: &ArrayBase<OwnedRepr<T>, Ix3>,
    axis: Axis,
) -> Array<usize, Ix2> {
    let num_rows = array.len_of(axis.opposite());
    let num_cols = array.len_of(axis);

    let mut indices = Array::<usize, Ix2>::zeros((num_rows, 1));

    for i in 0..num_rows {
        let mut max_value = array[(i, 0)];
        let mut max_index = 0;
        for j in 1..num_cols {
            let value = array[(i, j)];
            if value > max_value {
                max_value = value;
                max_index = j;
            }
        }
        indices[(i, 0)] = max_index;
    }

    indices
}

// Main struct that encapsulates all functions and data
pub struct OnnxModelRunner<'a> {
    pub session: Session<'a>,
}

impl<'a> OnnxModelRunner<'a> {
    pub fn new(model_path: &str) -> Result<Self, JsValue> {
        let environment = Environment::builder()
            .with_name("Diacriticizer")
            .with_execution_providers([ExecutionProvider::cuda()])
            .build()?;

        let session = SessionBuilder::new(&environment)?
            .with_optimization_level(GraphOptimizationLevel::Level1)?
            .with_intra_op_num_threads(1)?
            .with_model_from_file(model_path)?;

        Ok(Self { session })
    }

    pub fn process(&self, embeddings: Vec<Vec<i32>>) -> Result<Array<f32, Ix3>, JsValue> {
        let input_tensor: Array<i32, _> = Array::from_shape_vec(
            (embeddings.len(), embeddings[0].len()),
            embeddings.into_iter().flatten().collect(),
        )
        .unwrap();

        let outputs: Vec<DynOrtTensor<ndarray::Dim<ndarray::IxDynImpl>>> = self
            .session
            .run(vec![("input", InputTensor::from_array(input_tensor))])
            .map_err(|_| JsValue::from_str("Failed to run model"))?;

        let output_array: Array<f32, Ix3> = outputs[0]
            .try_extract()
            .unwrap()
            .try_into()
            .map_err(|_| JsValue::from_str("Failed to convert tensor to array"))?;
        Ok(output_array)
    }
}

#[wasm_bindgen]
pub struct Diacriticizer {
    onnx_model_runner: OnnxModelRunner<'static>,
    input_vectorizer: InputVectorizer,
    output_deembedder: StringMapper,
}

#[wasm_bindgen]
impl Diacriticizer {
    #[wasm_bindgen(constructor)]
    pub fn new(
        model_path: &str,
        input_vectorizer_json_path: &str,
        string_mapper_json_path: &str,
    ) -> Result<Diacriticizer, JsValue> {
        let onnx_model_runner = OnnxModelRunner::new(model_path)?;
        let input_vectorizer = InputVectorizer::new(input_vectorizer_json_path)?;
        let output_deembedder = StringMapper::new(string_mapper_json_path)?;
        Ok(Diacriticizer {
            onnx_model_runner,
            input_vectorizer,
            output_deembedder,
        })
    }

    #[wasm_bindgen]
    pub fn process(&self, input: &str) -> Result<String, JsValue> {
        let input_sentences = split_sentences(input);
        let input_embeddings = self.input_vectorizer.process(input_sentences)?;
        let logits = self.onnx_model_runner.process(input_embeddings)?;
        let output_diacritics = self.output_deembedder.process(logits);
        let output_string = merge_characters(input_sentences, output_diacritics);
        Ok(output_string)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diacriticizer() {
        let diacriticizer = Diacriticizer::new("model.onnx", "input.json", "output.json").unwrap();
        let input = "اللغة العربية";
        let output = diacriticizer.process(input).unwrap();
        assert_eq!(output, "اَللُّغَةُ اَلْعَرَبِيَّةُ");
    }
}
