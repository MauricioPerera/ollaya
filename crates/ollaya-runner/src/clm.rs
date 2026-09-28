//! ONNX Runtime engine for Contrastive-LM's CLM (layout `clm-v1`).
//!
//! Every question contributes one state text and one text per option (`ollaya_decision::clm`).
//! The graph maps `input_ids` [rows, seq] (one text per row, right-padded, positions implicit, no
//! mask: every layer is causal), `last_pos` [rows] and `action` [rows] (1 for an option text) to
//! each row's L2-normalised projection `z` [rows, 512]. An option's logit is `scale * z_option .
//! z_state`.
//!
//! Projections are cached by text and head, as upstream caches them: a question asked again, or
//! another question with the same options, costs only its state text.

use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::Mutex;

use ndarray::{Array1, Array2, Ix2};
use ollaya_decision::clm::{ClmLayout, ClmQuestion};
use ollaya_decision::{Calibration, CalibrationFile, Questions};
use ort::session::Session;
use serde::Deserialize;
use serde_json::Value;

use crate::decider::WeightsInMemory;
use crate::engine::Engine;
use crate::onnx::{CudaArena, Device, ModelFiles, load_tokenizer, session_for};
use crate::{Error, Output, QuestionOutput};

/// Rows per `session.run`: the export's row axis is 1..=4096.
const MAX_ROWS: usize = 4096;
/// Padded tokens per `session.run`, as for the other large decoders.
const TOKEN_BUDGET: usize = 8192;
/// Projections kept (512 floats each, 2 KiB): about 16 MiB.
const CACHE_ENTRIES: usize = 8192;
const INPUTS: [&str; 3] = ["input_ids", "last_pos", "action"];
const OUTPUT: &str = "z";

#[derive(Debug, Clone, Deserialize)]
struct SpecialTokens {
    pad: u32,
}

/// The fields of the `decision` layer this engine reads.
#[derive(Debug, Clone, Deserialize)]
struct DecisionConfig {
    engine: String,
    layout: String,
    special_tokens: SpecialTokens,
    #[serde(default)]
    weights_in_memory: WeightsInMemory,
    #[serde(flatten)]
    clm: ClmLayout,
}

/// A text and the head it goes through.
type Key = (String, bool);

/// A bounded map of projections, oldest out first.
#[derive(Default)]
struct Cache {
    map: HashMap<Key, Vec<f32>>,
    order: VecDeque<Key>,
}

impl Cache {
    fn get(&self, key: &Key) -> Option<&Vec<f32>> {
        self.map.get(key)
    }

    fn put(&mut self, key: Key, z: Vec<f32>) {
        if self.map.insert(key.clone(), z).is_none() {
            self.order.push_back(key);
            while self.order.len() > CACHE_ENTRIES {
                if let Some(old) = self.order.pop_front() {
                    self.map.remove(&old);
                }
            }
        }
    }
}

/// One text to embed: its token ids and head.
#[derive(Debug, Clone, PartialEq)]
pub struct ClmRow {
    pub ids: Vec<u32>,
    pub action: bool,
}

pub struct ClmModel {
    session: Mutex<Session>,
    tokenizer: tokenizers::Tokenizer,
    pad: u32,
    cache: Mutex<Cache>,
    pub layout: ClmLayout,
    pub calibration: Calibration,
    pub device: Device,
}

fn read_json<T: serde::de::DeserializeOwned>(path: &Path) -> Result<T, Error> {
    let text = std::fs::read_to_string(path)
        .map_err(|e| Error::Model(format!("{}: {e}", path.display())))?;
    serde_json::from_str(&text).map_err(|e| Error::Model(format!("{}: {e}", path.display())))
}

impl Engine for ClmModel {
    fn run(&self, state: &Value, questions: &Questions) -> Result<Output, Error> {
        let defs: Vec<(&str, &Value)> = questions
            .iter()
            .map(|(qid, q)| (qid.as_str(), &q.definition))
            .collect();
        self.answer(state, &defs)
    }
}

impl ClmModel {
    /// Load a model exported to one directory (development and parity tooling).
    pub fn load(dir: &Path, device: Device, intra_threads: Option<usize>) -> Result<Self, Error> {
        Self::load_files(&ModelFiles::dir(dir), device, intra_threads)
    }

    pub fn load_files(
        files: &ModelFiles,
        device: Device,
        intra_threads: Option<usize>,
    ) -> Result<Self, Error> {
        let config: DecisionConfig = read_json(&files.decision)?;
        if config.engine != "onnx" || config.layout != "clm-v1" {
            return Err(Error::Model(format!(
                "unsupported engine/layout {}/{}; this engine serves onnx/clm-v1",
                config.engine, config.layout
            )));
        }
        config
            .clm
            .validate()
            .map_err(|e| Error::Model(format!("{}: {e}", files.decision.display())))?;
        let calibration = match &files.calibration {
            Some(path) => Calibration::from_file(&read_json::<CalibrationFile>(path)?),
            None => Calibration::default(),
        };
        let tokenizer = load_tokenizer(&files.tokenizer)?;
        let weights = config.weights_in_memory;
        let session = session_for(
            &files.graph,
            device,
            intra_threads,
            CudaArena::SameAsRequested,
            |b| weights.configure(crate::decider::configure(b, device)?),
        )?;
        let inputs: Vec<&str> = session.inputs().iter().map(|i| i.name()).collect();
        if inputs.len() != INPUTS.len()
            || !INPUTS.iter().all(|n| inputs.contains(n))
            || !session.outputs().iter().any(|o| o.name() == OUTPUT)
        {
            return Err(Error::Model(format!(
                "graph inputs {inputs:?} do not match the clm contract ({INPUTS:?} -> {OUTPUT:?})"
            )));
        }
        Ok(ClmModel {
            session: Mutex::new(session),
            tokenizer,
            pad: config.special_tokens.pad,
            cache: Mutex::new(Cache::default()),
            layout: config.clm,
            calibration,
            device,
        })
    }

    /// Forget every cached projection (benchmarks).
    pub fn clear_cache(&self) {
        *self.cache.lock().expect("cache mutex poisoned") = Cache::default();
    }

    /// Every question's texts, in request order.
    pub fn questions(
        &self,
        state: &Value,
        defs: &[(&str, &Value)],
    ) -> Result<Vec<ClmQuestion>, Error> {
        defs.iter()
            .map(|(qid, def)| Ok(self.layout.question(state, qid, def)?))
            .collect()
    }

    /// A text's token ids, as upstream's tokenizer call makes them (Qwen3 adds no special
    /// tokens). An empty text is embedded as " ", as upstream's training recipe does.
    pub fn row(&self, text: &str, action: bool) -> Result<ClmRow, Error> {
        let encode = |t: &str| {
            self.tokenizer
                .encode_fast(t, false)
                .map(|e| e.get_ids().to_vec())
                .map_err(|e| Error::Decision(ollaya_decision::Error::Tokenizer(e.to_string())))
        };
        let mut ids = encode(text)?;
        if ids.is_empty() {
            ids = encode(" ")?;
        }
        if ids.len() > self.layout.max_text_tokens {
            return Err(Error::Decision(ollaya_decision::Error::invalid(format!(
                "a {} text is {} tokens; the model reads up to {}",
                if action { "option" } else { "state" },
                ids.len(),
                self.layout.max_text_tokens
            ))));
        }
        Ok(ClmRow { ids, action })
    }

    /// Answer the questions: one logit per option.
    pub fn answer(&self, state: &Value, defs: &[(&str, &Value)]) -> Result<Output, Error> {
        let questions = self.questions(state, defs)?;
        let mut texts: Vec<Key> = Vec::new();
        for q in &questions {
            texts.push((q.state.clone(), false));
            texts.extend(q.options.iter().map(|o| (o.clone(), true)));
        }
        let (z, input_tokens) = self.project(&texts)?;
        let dot = |a: &[f32], b: &[f32]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>();
        let outputs = questions
            .iter()
            .map(|q| {
                let zs = &z[&(q.state.clone(), false)];
                QuestionOutput {
                    logits: q
                        .options
                        .iter()
                        .map(|o| self.layout.scale * dot(&z[&(o.clone(), true)], zs))
                        .collect(),
                    act_logits: None,
                }
            })
            .collect();
        Ok(Output {
            questions: outputs,
            input_tokens,
            state_tokens: 0,
            state_truncated: false,
        })
    }

    /// Projections of `texts` (duplicates computed once, cached ones not at all), and the tokens
    /// run through the encoder.
    pub fn project(&self, texts: &[Key]) -> Result<(HashMap<Key, Vec<f32>>, usize), Error> {
        let mut out: HashMap<Key, Vec<f32>> = HashMap::new();
        let mut todo: Vec<(Key, ClmRow)> = Vec::new();
        {
            let cache = self.cache.lock().expect("cache mutex poisoned");
            for key in texts {
                if out.contains_key(key) || todo.iter().any(|(k, _)| k == key) {
                    continue;
                }
                match cache.get(key) {
                    Some(z) => {
                        out.insert(key.clone(), z.clone());
                    }
                    None => todo.push((key.clone(), self.row(&key.0, key.1)?)),
                }
            }
        }
        let rows: Vec<ClmRow> = todo.iter().map(|(_, r)| r.clone()).collect();
        let tokens = rows.iter().map(|r| r.ids.len()).sum();
        let z = self.embed(&rows)?;
        let mut cache = self.cache.lock().expect("cache mutex poisoned");
        for ((key, _), z) in todo.into_iter().zip(z) {
            cache.put(key.clone(), z.clone());
            out.insert(key, z);
        }
        Ok((out, tokens))
    }

    /// Run the graph on `rows`, shortest first in token-budgeted batches: each row's projection.
    pub fn embed(&self, rows: &[ClmRow]) -> Result<Vec<Vec<f32>>, Error> {
        let mut order: Vec<usize> = (0..rows.len()).collect();
        order.sort_by_key(|&i| rows[i].ids.len());
        let lens: Vec<usize> = order.iter().map(|&i| rows[i].ids.len()).collect();
        let mut z = vec![Vec::new(); rows.len()];
        let mut session = self.session.lock().expect("session mutex poisoned");
        for range in crate::engine::batches(&lens, TOKEN_BUDGET, MAX_ROWS) {
            let batch = &order[range.clone()];
            // The export's sequence axis starts at 2.
            let seq = lens[range].iter().copied().max().unwrap_or(0).max(2);
            let mut input_ids = Array2::<i64>::from_elem((batch.len(), seq), i64::from(self.pad));
            let mut last_pos = Array1::<i64>::zeros(batch.len());
            let mut action = Array1::<i64>::zeros(batch.len());
            for (r, &i) in batch.iter().enumerate() {
                for (c, &id) in rows[i].ids.iter().enumerate() {
                    input_ids[[r, c]] = i64::from(id);
                }
                last_pos[r] = rows[i].ids.len() as i64 - 1;
                action[r] = i64::from(rows[i].action);
            }
            let outputs = session.run(ort::inputs![
                "input_ids" => ort::value::Tensor::from_array(input_ids)?,
                "last_pos" => ort::value::Tensor::from_array(last_pos)?,
                "action" => ort::value::Tensor::from_array(action)?,
            ])?;
            let out = outputs[OUTPUT]
                .try_extract_array::<f32>()?
                .into_dimensionality::<Ix2>()
                .map_err(|e| Error::Model(format!("{OUTPUT}: {e}")))?;
            if out.nrows() != batch.len() {
                return Err(Error::Model(format!(
                    "{OUTPUT} has {} rows for {} texts",
                    out.nrows(),
                    batch.len()
                )));
            }
            for (&i, row) in batch.iter().zip(out.rows()) {
                z[i] = row.to_vec();
            }
        }
        Ok(z)
    }
}
