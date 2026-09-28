//! `clm-v1`: Contrastive-LM's CLM (a Qwen3-8B encoder with a state head and an action head), whose
//! texts follow upstream `clm.schema.build_pairs` (`ollaya_convert.families.clm.ref`).
//!
//! A question becomes one state text and one text per option, each embedded on its own:
//!
//! ```text
//! state   to_text(state).strip() + "\n\n" + to_text(instructions).strip()   (either alone if the other is empty)
//! choice  to_text(description), or the label when the description is null or ""
//! score   to_text(level)
//! noul    "false: " + (description, or "No. This is false: " + instructions, or "false")
//!         "true: "  + (description, or "Yes. This is true: " + instructions, or "true")
//! ```
//!
//! The graph embeds each text (Qwen3-8B's last token, L2-normalised) and projects it through the
//! state head or the action head; an option's logit is `scale * z_option . z_state`.

use std::fmt::Write;

use serde::Deserialize;
use serde_json::{Number, Value};

use crate::{Error, pyjson, pyrepr};

/// The layout as `decision.json` declares it.
#[derive(Debug, Clone, Deserialize)]
pub struct ClmLayout {
    /// A longer text is rejected (upstream cuts it to its last tokens instead).
    pub max_text_tokens: usize,
    /// `min(exp(logit_scale), 100)`: option logits are `scale * cosine`.
    pub scale: f32,
}

/// One question's texts: the state text, and the option labels and texts in answer order.
#[derive(Debug, Clone, PartialEq)]
pub struct ClmQuestion {
    pub state: String,
    pub keys: Vec<String>,
    pub options: Vec<String>,
}

impl ClmLayout {
    /// Reject configurations this layout cannot run.
    pub fn validate(&self) -> Result<(), Error> {
        if self.max_text_tokens == 0 || !(self.scale.is_finite() && self.scale > 0.0) {
            return Err(Error::invalid(format!(
                "max_text_tokens={} must be positive and scale={} finite and positive",
                self.max_text_tokens, self.scale
            )));
        }
        Ok(())
    }

    /// `build_pairs` for one question, validated as upstream's `candidates` does. `def` is the
    /// definition as the caller sent it: upstream needs no `instructions`, unlike the typed
    /// questions the other layouts parse.
    pub fn question(&self, state: &Value, qid: &str, def: &Value) -> Result<ClmQuestion, Error> {
        let bad = |msg: &str| Error::invalid(format!("question {qid:?}: {msg}"));
        if !def.is_object() {
            return Err(bad(
                "a question is an object with type, instructions and criteria",
            ));
        }
        let instructions = def.get("instructions").unwrap_or(&Value::Null);
        let crit = def.get("criteria").unwrap_or(&Value::Null);
        let (keys, options) = match def.get("type").and_then(Value::as_str) {
            Some("choice") => {
                let m = crit.as_object().filter(|m| !m.is_empty()).ok_or_else(|| {
                    bad(
                        "this model takes choice criteria as a non-empty object of label -> \
                         description; write a list of labels as {\"label\": null, ...}",
                    )
                })?;
                let keys: Vec<String> = m.keys().cloned().collect();
                let options = m
                    .iter()
                    .map(|(k, d)| match d {
                        Value::Null => k.clone(),
                        Value::String(s) if s.is_empty() => k.clone(),
                        d => to_text(d),
                    })
                    .collect();
                (keys, options)
            }
            Some("score") => {
                let levels = crit.as_array().filter(|l| l.len() >= 2).ok_or_else(|| {
                    bad("this model takes score criteria as an ordered list of at least 2 levels")
                })?;
                (
                    (0..levels.len()).map(|i| i.to_string()).collect(),
                    levels.iter().map(to_text).collect(),
                )
            }
            Some("noul") => {
                let ins = pyrepr::strip(&to_text(instructions)).to_owned();
                let side = |key: &str, default: &str| {
                    let d = crit.as_object().and_then(|m| m.get(key));
                    let text = match d {
                        None | Some(Value::Null) => None,
                        Some(Value::String(s)) if s.is_empty() => None,
                        Some(d) => Some(to_text(d)),
                    };
                    let text = text.unwrap_or_else(|| {
                        if ins.is_empty() {
                            key.to_owned()
                        } else {
                            format!("{default}{ins}")
                        }
                    });
                    format!("{key}: {text}")
                };
                (
                    vec!["false".into(), "true".into()],
                    vec![
                        side("false", "No. This is false: "),
                        side("true", "Yes. This is true: "),
                    ],
                )
            }
            _ => return Err(bad("type must be noul, choice or score")),
        };
        Ok(ClmQuestion {
            state: state_text(state, instructions),
            keys,
            options,
        })
    }
}

/// `state_text`: the context first, the question last, after a blank line.
pub fn state_text(state: &Value, instructions: &Value) -> String {
    let (s, i) = (to_text(state), to_text(instructions));
    let (s, i) = (pyrepr::strip(&s), pyrepr::strip(&i));
    match (s.is_empty(), i.is_empty()) {
        (false, false) => format!("{s}\n\n{i}"),
        (false, true) => s.to_owned(),
        _ => i.to_owned(),
    }
}

/// `clm.schema.to_text`: strings as they are; `true`/`false`; numbers as Python's `str()`; an
/// object as `key: value` fields (top-level fields separated by a blank line, nested ones indented
/// by two spaces), an array as one `- item` line per element.
pub fn to_text(v: &Value) -> String {
    to_text_at(v, 0)
}

fn nonempty_container(v: &Value) -> bool {
    match v {
        Value::Object(m) => !m.is_empty(),
        Value::Array(a) => !a.is_empty(),
        _ => false,
    }
}

fn to_text_at(v: &Value, indent: usize) -> String {
    let pad = " ".repeat(indent);
    match v {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        Value::Bool(b) => if *b { "true" } else { "false" }.into(),
        Value::Number(n) => py_number(n),
        Value::Object(m) => {
            let sep = if indent == 0 { "\n\n" } else { "\n" };
            let mut out = String::new();
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(sep);
                }
                if nonempty_container(x) {
                    let _ = write!(out, "{pad}{k}:\n{}", to_text_at(x, indent + 2));
                } else {
                    let _ = write!(out, "{pad}{k}: {}", to_text_at(x, 0));
                }
            }
            out
        }
        Value::Array(items) => {
            let mut out = String::new();
            for (i, x) in items.iter().enumerate() {
                if i > 0 {
                    out.push('\n');
                }
                if nonempty_container(x) {
                    let _ = write!(out, "{pad}-\n{}", to_text_at(x, indent + 2));
                } else {
                    let _ = write!(out, "{pad}- {}", to_text_at(x, 0));
                }
            }
            out
        }
    }
}

/// Python's `str()` of a JSON number: integers as written, floats as `repr`.
fn py_number(n: &Number) -> String {
    if let Some(i) = n.as_i64() {
        i.to_string()
    } else if let Some(u) = n.as_u64() {
        u.to_string()
    } else {
        pyjson::float_repr(n.as_f64().unwrap_or(f64::NAN))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn layout() -> ClmLayout {
        ClmLayout {
            max_text_tokens: 2048,
            scale: 100.0,
        }
    }

    fn one(state: Value, q: Value) -> Result<ClmQuestion, Error> {
        layout().question(&state, "q", &q)
    }

    #[test]
    fn texts_follow_upstream_schema() {
        let q = one(
            json!("Customer: my invoice was charged twice!"),
            json!({"type": "choice", "instructions": "Which team?",
                   "criteria": {"billing": "Charges, invoices, refunds", "other": null, "x": ""}}),
        )
        .unwrap();
        assert_eq!(
            q.state,
            "Customer: my invoice was charged twice!\n\nWhich team?"
        );
        assert_eq!(q.keys, ["billing", "other", "x"]);
        assert_eq!(q.options, ["Charges, invoices, refunds", "other", "x"]);

        let q = one(
            json!(""),
            json!({"type": "noul", "instructions": " Is it urgent? "}),
        )
        .unwrap();
        assert_eq!(q.state, "Is it urgent?");
        assert_eq!(
            q.options,
            [
                "false: No. This is false: Is it urgent?",
                "true: Yes. This is true: Is it urgent?"
            ]
        );
        let q = one(
            json!("x"),
            json!({"type": "noul", "criteria": {"true": "a refund"}}),
        )
        .unwrap();
        assert_eq!(q.options, ["false: false", "true: a refund"]);

        let q = one(
            json!("x"),
            json!({"type": "score", "criteria": ["Calm", {"level": 2}]}),
        )
        .unwrap();
        assert_eq!(q.keys, ["0", "1"]);
        assert_eq!(q.options, ["Calm", "level: 2"]);
    }

    #[test]
    fn upstream_rejections() {
        assert!(
            one(
                json!("x"),
                json!({"type": "choice", "criteria": ["a", "b"]})
            )
            .is_err()
        );
        assert!(
            one(
                json!("x"),
                json!({"type": "score", "criteria": {"0": "a", "1": "b"}})
            )
            .is_err()
        );
    }

    #[test]
    fn json_states_render_as_prose() {
        let v = json!({"subject": "Refund", "order": {"id": 7, "paid": true, "amount": 1.0,
                       "items": ["a", {"sku": "x"}], "tags": []}, "none": null});
        assert_eq!(
            to_text(&v),
            "subject: Refund\n\norder:\n  id: 7\n  paid: true\n  amount: 1.0\n  items:\n    - a\n    -\n      sku: x\n  tags: \n\nnone: "
        );
    }
}
