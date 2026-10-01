//! `cygnet-v1`: Cygnet, frozen `google/gemma-4-12B-it` with a one-token option-letter readout
//! (`docs/families/cygnet.md`). A port of `shim/cygnet_shim.py` (`SYSTEM`, `build_prompt`) and
//! `shim/decision_server.py` (`parse_question`) in github.com/blockbrain-ai/cygnet-recipe, following
//! `convert/ollaya_convert/families/cygnet/ref.py`:
//!
//! ```text
//! user = rstrip(state_text) + "\n\n" + rstrip(instructions) + "\n\nOptions:\n"
//!        + "A. text_0\nB. text_1\n..." + "\n\nAnswer with the letter of exactly one option, and nothing else:"
//! ids  = [BOS] ⧺ pre ⧺ tok(SYSTEM) ⧺ mid ⧺ tok(user) ⧺ post
//! ```
//!
//! `state_text` is the state itself when it is a string, nothing when it is empty or falsy, and
//! otherwise `json.dumps(indent=1, ensure_ascii=False)`, the rendering Cygnet was measured with.
//! Option texts: a choice's description (its label when blank, `"label: <json>"` for JSON); a score
//! level (`"Level i"` when blank); a noul reads `false` (A) then `true` (B), `"No"` and `"Yes"` when
//! undescribed. The template pieces are Gemma's chat template rendered with thinking off
//! (`decision.json`); specials are parsed only there. The option logits are the letters' next-token
//! logits; Cygnet's temperature (3.4) calibrates them. One pass reads up to 20 options; the
//! upstream server's grouped reading of wider questions is not ported.

use serde::Deserialize;
use serde_json::Value;

use crate::question::QType;
use crate::{Error, pyjson};

pub const LAYOUT: &str = "cygnet-v1";

/// `cygnet_shim.SYSTEM`, verbatim (its dash is U+2014).
pub const SYSTEM: &str = "You are a calibration engine. You never answer in prose. You are given a \
state, a question and a numbered set of options, and you choose exactly one option. You reply with \
that option's LETTER and nothing else \u{2014} a single character, no words, no punctuation, no \
explanation.";
pub const ANSWER_LINE: &str = "Answer with the letter of exactly one option, and nothing else:";
const MAX_SCORE_LEVELS: usize = 10;

#[derive(Debug, Clone, Deserialize)]
pub struct Template {
    pub pre: String,
    pub mid: String,
    pub post: String,
}

/// `decision.json` of a `cygnet-v1` model (the fields the runtime reads).
#[derive(Debug, Clone, Deserialize)]
pub struct CygnetConfig {
    pub layout: String,
    /// Gemma's chat template around the system and user messages, thinking off.
    pub template: Template,
    /// Whether the GGUF adds a BOS the template did not write.
    pub add_bos: bool,
    /// `A`..`Z` and their single tokens.
    pub labels: crate::llm_logits::LabelTable,
    /// Options one pass reads (the upstream group size).
    pub max_options: usize,
}

/// One question as the model reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct QuestionPrompt {
    pub qtype: QType,
    pub user: String,
    pub label_ids: Vec<u32>,
    /// Prompt position of each wire option (noul: `false` is `A`, as on the wire).
    pub wire_order: Vec<usize>,
}

/// Python's `str.isspace`.
fn py_isspace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// Python truthiness of a JSON value.
fn truthy(v: &Value) -> bool {
    match v {
        Value::Null => false,
        Value::Bool(b) => *b,
        Value::Number(n) => n.as_f64() != Some(0.0),
        Value::String(s) => !s.is_empty(),
        Value::Array(a) => !a.is_empty(),
        Value::Object(o) => !o.is_empty(),
    }
}

/// `_blank`: null, or a string of whitespace only.
fn blank(v: &Value) -> bool {
    match v {
        Value::Null => true,
        Value::String(s) => s.chars().all(py_isspace),
        _ => false,
    }
}

/// `_text`: a string as it is, anything else as `json.dumps(ensure_ascii=False)`.
fn text(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        v => pyjson::dumps(v, false),
    }
}

/// `body.get("state") or ""`, then `build_prompt`'s rendering.
pub fn state_text(state: &Value) -> String {
    match state {
        v if !truthy(v) => String::new(),
        Value::String(s) => s.clone(),
        v => pyjson::dumps_indent(v, false, 1),
    }
}

/// The user message for option texts in prompt order.
pub fn user_message(state: &Value, instructions: &Value, options: &[String]) -> String {
    let instructions = match instructions {
        v if !truthy(v) => String::new(),
        v => text(v),
    };
    let mut lines = vec![
        state_text(state).trim_end_matches(py_isspace).to_owned(),
        String::new(),
        instructions.trim_end_matches(py_isspace).to_owned(),
        String::new(),
        "Options:".to_owned(),
    ];
    lines.extend(
        options
            .iter()
            .zip('A'..='Z')
            .map(|(o, l)| format!("{l}. {o}")),
    );
    lines.push(String::new());
    lines.push(ANSWER_LINE.to_owned());
    lines.join("\n")
}

impl CygnetConfig {
    pub fn validate(&self) -> Result<(), Error> {
        let bad = |msg: String| Err(Error::invalid(format!("decision.json: {msg}")));
        if self.layout != LAYOUT {
            return bad(format!("layout {:?} is not {LAYOUT}", self.layout));
        }
        let letters: Vec<String> = ('A'..='Z').map(String::from).collect();
        if self.labels.strings != letters || self.labels.ids.len() != 26 {
            return bad("labels must be A..Z with one token each".into());
        }
        if !(2..=26).contains(&self.max_options) {
            return bad(format!("max_options {} must be 2..=26", self.max_options));
        }
        Ok(())
    }

    /// Every question's prompt, in request order. The request is rejected as a whole when any
    /// question is.
    pub fn questions(
        &self,
        state: &Value,
        questions: &Value,
    ) -> Result<Vec<(String, QuestionPrompt)>, Error> {
        let qs = questions
            .as_object()
            .filter(|q| !q.is_empty())
            .ok_or_else(|| {
                Error::invalid("questions must be a non-empty object of named questions")
            })?;
        qs.iter()
            .map(|(qid, src)| Ok((qid.clone(), self.question(qid, src, state)?)))
            .collect()
    }

    fn question(&self, qid: &str, src: &Value, state: &Value) -> Result<QuestionPrompt, Error> {
        let bad = |msg: &str| Error::invalid(format!("question {qid:?}: {msg}"));
        let src = src.as_object().ok_or_else(|| bad("must be an object"))?;
        let described = |v: &Value| {
            matches!(
                v,
                Value::Null | Value::String(_) | Value::Object(_) | Value::Array(_)
            )
        };
        let crit = src.get("criteria").unwrap_or(&Value::Null);
        let (qtype, options): (QType, Vec<String>) = match src.get("type").and_then(Value::as_str) {
            Some("choice") => {
                let pairs: Vec<(String, Value)> = match crit {
                    Value::Object(m) if !m.is_empty() => {
                        m.iter().map(|(k, v)| (k.clone(), v.clone())).collect()
                    }
                    // A list of labels: {label: null}, each label once.
                    Value::Array(items) if !items.is_empty() => {
                        let mut out: Vec<(String, Value)> = Vec::new();
                        for item in items {
                            let k = item
                                .as_str()
                                .ok_or_else(|| bad("choice labels must be strings"))?;
                            if !out.iter().any(|(p, _)| p == k) {
                                out.push((k.to_owned(), Value::Null));
                            }
                        }
                        out
                    }
                    _ => return Err(bad("criteria must be a non-empty object of options")),
                };
                let mut options = Vec::with_capacity(pairs.len());
                for (label, d) in &pairs {
                    if !described(d) {
                        return Err(bad("every option description must be text, JSON or null"));
                    }
                    options.push(match d {
                        d if blank(d) => label.clone(),
                        Value::String(s) => s.clone(),
                        d => format!("{label}: {}", text(d)),
                    });
                }
                (QType::Choice, options)
            }
            Some("score") => {
                let levels = crit
                    .as_array()
                    .filter(|l| !l.is_empty())
                    .ok_or_else(|| bad("criteria must be a non-empty list of levels"))?;
                if levels.len() > MAX_SCORE_LEVELS {
                    return Err(bad(&format!(
                        "{} levels; a score takes at most {MAX_SCORE_LEVELS}",
                        levels.len()
                    )));
                }
                let mut options = Vec::with_capacity(levels.len());
                for (i, d) in levels.iter().enumerate() {
                    if !described(d) {
                        return Err(bad("every level must be text, JSON or null"));
                    }
                    options.push(if blank(d) {
                        format!("Level {i}")
                    } else {
                        text(d)
                    });
                }
                (QType::Score, options)
            }
            Some("noul") => {
                let empty = serde_json::Map::new();
                let m = match crit {
                    Value::Null => &empty,
                    Value::Object(m) => m,
                    _ => {
                        return Err(bad(
                            "criteria must be an object with 'true' and 'false' descriptions",
                        ));
                    }
                };
                let mut sides: [Option<&Value>; 2] = [None, None];
                for (key, d) in m {
                    let slot = match key.to_lowercase().as_str() {
                        "false" => 0,
                        "true" => 1,
                        _ => return Err(bad("criteria takes only 'true' and 'false', once each")),
                    };
                    if sides[slot].is_some() {
                        return Err(bad("criteria takes only 'true' and 'false', once each"));
                    }
                    if !described(d) {
                        return Err(bad("every description must be text, JSON or null"));
                    }
                    sides[slot] = Some(d);
                }
                let side = |i: usize, default: &str| match sides[i] {
                    Some(d) if !blank(d) => text(d),
                    _ => default.to_owned(),
                };
                (QType::Noul, vec![side(0, "No"), side(1, "Yes")])
            }
            _ => return Err(bad("type must be 'choice', 'score' or 'noul'")),
        };
        if options.len() > self.max_options {
            return Err(Error::TooManyOptions {
                question: qid.to_owned(),
                options: options.len(),
                head_max_len: self.max_options,
            });
        }
        let instructions = src.get("instructions").unwrap_or(&Value::Null);
        let n = options.len();
        Ok(QuestionPrompt {
            qtype,
            user: user_message(state, instructions, &options),
            label_ids: self.labels.ids[..n].to_vec(),
            wire_order: (0..n).collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn config() -> CygnetConfig {
        CygnetConfig {
            layout: LAYOUT.into(),
            template: Template {
                pre: "<bos><|turn>system\n".into(),
                mid: "<turn|>\n<|turn>user\n".into(),
                post: "<turn|>\n<|turn>model\n".into(),
            },
            add_bos: false,
            labels: crate::llm_logits::LabelTable {
                strings: ('A'..='Z').map(String::from).collect(),
                ids: (200..226).collect(),
            },
            max_options: 20,
        }
    }

    #[test]
    fn builds_the_shims_prompt() {
        let c = config();
        let qs = c
            .questions(
                &json!({"msg": "Zoë", "n": [1, 2]}),
                &json!({
                    "n": {"type": "noul", "instructions": "Refund?  ", "criteria": {"True": {"why": 1}}},
                    "c": {"type": "choice", "instructions": {"q": "Team?"}, "criteria": {"billing": "cards", "other": " ", "x": [1]}},
                    "s": {"type": "score", "criteria": [null, "bad"]},
                }),
            )
            .unwrap();
        assert_eq!(
            qs[0].1.user,
            "{\n \"msg\": \"Zoë\",\n \"n\": [\n  1,\n  2\n ]\n}\n\nRefund?\n\nOptions:\nA. No\nB. {\"why\": 1}\n\n\
             Answer with the letter of exactly one option, and nothing else:"
        );
        assert_eq!(qs[0].1.wire_order, [0, 1]);
        assert!(
            qs[1]
                .1
                .user
                .contains("\n\n{\"q\": \"Team?\"}\n\nOptions:\nA. cards\nB. other\nC. x: [1]\n\n")
        );
        assert!(
            qs[2]
                .1
                .user
                .contains("\n\n\n\nOptions:\nA. Level 0\nB. bad\n\n")
        );
        assert_eq!(qs[2].1.label_ids, [200, 201]);
        assert_eq!(state_text(&json!({})), "");
        assert_eq!(state_text(&json!("  hi  ")), "  hi  ");
    }

    #[test]
    fn rejects_what_the_server_rejects() {
        let c = config();
        for q in [
            json!({"type": "noul", "instructions": "x", "criteria": {"yes": "y"}}),
            json!({"type": "noul", "instructions": "x", "criteria": {"true": "a", "TRUE": "b"}}),
            json!({"type": "choice", "instructions": "x", "criteria": {"a": 1}}),
            json!({"type": "choice", "instructions": "x", "criteria": {}}),
            json!({"type": "score", "instructions": "x", "criteria": {"a": 1}}),
            json!({"type": "maybe", "instructions": "x"}),
        ] {
            assert!(
                matches!(
                    c.questions(&json!("s"), &json!({"q": q})),
                    Err(Error::Invalid(_))
                ),
                "{q}"
            );
        }
        let wide: Vec<String> = (0..21).map(|i| format!("o{i}")).collect();
        assert!(matches!(
            c.questions(
                &json!("s"),
                &json!({"q": {"type": "choice", "instructions": "x", "criteria": wide}})
            ),
            Err(Error::TooManyOptions { options: 21, .. })
        ));
    }
}
