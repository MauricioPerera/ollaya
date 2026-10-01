//! `jebadiah-v1`: Jebadiah's prompt, which is AINode's own `/v1/systemone` renderer
//! (`docs/families/jebadiah.md`). A port of `scripts/jebadiah_prompt.py` and
//! `scripts/ainode_prompt_verbatim.py` in the author's GGUF repositories (github.com/getainode/ainode
//! at `e5c08938`), following `convert/ollaya_convert/families/jebadiah/ref.py`:
//!
//! ```text
//! user   = "STATE:\n" + serialize_state(state) + "\n\nQUESTION: " + instructions.strip()
//!          + "\n\nOPTIONS:\n" + "A. opt_0\nB. opt_1\n..." + "\n\n" + ANSWER_INSTRUCTION
//! pre    = "<|im_start|>system\n" + SYSTEM + "<|im_end|>\n<|im_start|>user\n"
//! post   = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
//! ids    = tok(pre, parse_special) ⧺ tok(user) ⧺ tok(post, parse_special)
//! ```
//!
//! `serialize_state` is a string verbatim, `null` as nothing, anything else compact JSON with sorted
//! keys. Options are `true` then `false` for a noul, a choice's keys in order, a score's levels,
//! each `"name: description"` with the description on one line, or the bare name. Labels are
//! `A`..`Z`, `AA`, `AB`, ... as long as each is one token. The author tokenizes the whole prompt with
//! special parsing; Ollaya parses specials only in its own template pieces, as `jevk5-v1` does. The
//! author cuts a state whose prompt passes 2,048 tokens; Ollaya rejects that question instead.

use serde::Deserialize;
use serde_json::Value;

use crate::question::QType;
use crate::{Error, pyjson};

pub const LAYOUT: &str = "jebadiah-v1";

pub const SYSTEM: &str = "You are a decision function. Answer with the single letter of the best option and nothing else.";
pub const ANSWER_INSTRUCTION: &str = "Answer with the label of one option and nothing else.";
pub const POST: &str = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
const MIN_SCORE_LEVELS: usize = 2;
const MAX_SCORE_LEVELS: usize = 10;
/// TypeSafe's option limit.
const MAX_OPTIONS: usize = 255;

/// `decision.json` of a `jebadiah-v1` model (the fields the runtime reads).
#[derive(Debug, Clone, Deserialize)]
pub struct JebadiahConfig {
    pub layout: String,
    /// `A`, `B`, ..., as far as each is one token, and their tokens.
    pub labels: crate::llm_logits::LabelTable,
    /// The author's extended alphabet for a question wider than `labels`: `A`..`Z`, then every
    /// two-letter uppercase string that is one token, in alphabetical order.
    pub extended_labels: crate::llm_logits::LabelTable,
    /// A longer prompt is rejected (the author cuts the state instead).
    pub max_prompt_tokens: usize,
}

/// One question as the model reads it.
#[derive(Debug, Clone, PartialEq)]
pub struct QuestionPrompt {
    pub qtype: QType,
    /// The user message; the prompt is `pre() + user + POST`.
    pub user: String,
    /// The labels' tokens, in prompt order.
    pub label_ids: Vec<u32>,
    /// Prompt position of each wire option (noul: `[1, 0]`, since `true` is `A`).
    pub wire_order: Vec<usize>,
}

/// AINode's label for option `index`: bijective base 26 (`A`..`Z`, `AA`, `AB`, ...).
pub fn option_label(index: usize) -> String {
    let mut n = index + 1;
    let mut out = Vec::new();
    while n > 0 {
        let rem = (n - 1) % 26;
        n = (n - 1) / 26;
        out.push(b'A' + rem as u8);
    }
    out.reverse();
    String::from_utf8(out).expect("ASCII")
}

/// Python's `str.isspace`.
fn py_isspace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// `str.strip()`.
fn strip(s: &str) -> &str {
    s.trim_matches(py_isspace)
}

/// The state as the model reads it.
pub fn serialize_state(state: &Value) -> String {
    match state {
        Value::Null => String::new(),
        Value::String(s) => s.clone(),
        v => pyjson::dumps_canonical(v),
    }
}

/// `option_text`: the name, then its description on one line.
fn option_text(name: &str, description: Option<&str>) -> String {
    match description {
        Some(d) if !strip(d).is_empty() => {
            let flat: Vec<&str> = d.split(py_isspace).filter(|w| !w.is_empty()).collect();
            format!("{name}: {}", flat.join(" "))
        }
        _ => name.to_owned(),
    }
}

/// The chat template before the user message: tokenized with special parsing.
pub fn pre() -> String {
    format!("<|im_start|>system\n{SYSTEM}<|im_end|>\n<|im_start|>user\n")
}

/// The user message for a question and its option texts in prompt order, labelled `A`, `B`, ...
/// or with `letters` (the extended alphabet).
pub fn user_message(
    state: &Value,
    question: &str,
    options: &[String],
    letters: Option<&[String]>,
) -> String {
    let mut lines = vec![
        "STATE:".to_owned(),
        serialize_state(state),
        String::new(),
        format!("QUESTION: {question}"),
        String::new(),
        "OPTIONS:".to_owned(),
    ];
    lines.extend(options.iter().enumerate().map(|(i, o)| match letters {
        Some(l) => format!("{}. {o}", l[i]),
        None => format!("{}. {o}", option_label(i)),
    }));
    lines.push(String::new());
    lines.push(ANSWER_INSTRUCTION.to_owned());
    lines.join("\n")
}

/// The prompt's token ids: [`pre`] and [`POST`] with special-token parsing, the user message
/// without it. `tokenize(text, parse_special)` adds no BOS.
pub fn token_ids<T, E>(
    user: &str,
    mut tokenize: impl FnMut(&str, bool) -> Result<Vec<T>, E>,
) -> Result<Vec<T>, E> {
    let mut ids = tokenize(&pre(), true)?;
    ids.extend(tokenize(user, false)?);
    ids.extend(tokenize(POST, true)?);
    Ok(ids)
}

impl JebadiahConfig {
    pub fn validate(&self) -> Result<(), Error> {
        let bad = |msg: String| Err(Error::invalid(format!("decision.json: {msg}")));
        if self.layout != LAYOUT {
            return bad(format!("layout {:?} is not {LAYOUT}", self.layout));
        }
        let l = &self.labels;
        if l.strings.len() < 26
            || l.strings.len() != l.ids.len()
            || l.strings
                .iter()
                .enumerate()
                .any(|(i, s)| *s != option_label(i))
        {
            return bad("labels must be A, B, ... with one token each".into());
        }
        let e = &self.extended_labels;
        if e.strings.len() != e.ids.len() || e.strings.iter().take(26).ne(l.strings.iter().take(26))
        {
            return bad("extended_labels must start with A..Z and give one token each".into());
        }
        if self.max_prompt_tokens == 0 {
            return bad("max_prompt_tokens must be positive".into());
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
                Error::invalid("'questions' must be a non-empty object of {id: question}")
            })?;
        qs.iter()
            .map(|(qid, src)| {
                if strip(qid).is_empty() {
                    return Err(Error::invalid(
                        "every question id must be a non-empty string",
                    ));
                }
                Ok((qid.clone(), self.question(qid, src, state)?))
            })
            .collect()
    }

    fn question(&self, qid: &str, src: &Value, state: &Value) -> Result<QuestionPrompt, Error> {
        let bad = |msg: &str| Error::invalid(format!("question {qid:?}: {msg}"));
        let src = src.as_object().ok_or_else(|| bad("must be an object"))?;
        let question = match src.get("instructions") {
            Some(Value::String(s)) if !strip(s).is_empty() => strip(s).to_owned(),
            _ => return Err(bad("needs a non-empty 'instructions' string")),
        };
        let crit = src.get("criteria").filter(|c| !c.is_null());
        // A criteria object's (stripped name, description), validated as AINode's criteria_pairs.
        let pairs =
            |m: &serde_json::Map<String, Value>| -> Result<Vec<(String, Option<String>)>, Error> {
                if m.is_empty() {
                    return Err(bad("needs a non-empty 'criteria' object"));
                }
                m.iter()
                    .map(|(name, d)| {
                        if strip(name).is_empty() {
                            return Err(bad("every 'criteria' name must be a non-empty string"));
                        }
                        let d = match d {
                            Value::Null => None,
                            Value::String(s) => Some(s.clone()),
                            _ => return Err(bad("every 'criteria' description must be a string")),
                        };
                        Ok((strip(name).to_owned(), d))
                    })
                    .collect()
            };
        let (qtype, options, wire_order): (QType, Vec<String>, Vec<usize>) = match src
            .get("type")
            .and_then(Value::as_str)
        {
            Some("noul") => {
                let mut described: [Option<String>; 2] = [None, None];
                if let Some(c) = crit {
                    let m = c.as_object().ok_or_else(|| {
                        bad("'criteria' must be an object of {true: ..., false: ...}")
                    })?;
                    for (name, d) in m {
                        let slot = match strip(name) {
                            "true" => 0,
                            "false" => 1,
                            _ => {
                                return Err(bad(
                                    "a noul's 'criteria' names only 'true' and 'false'",
                                ));
                            }
                        };
                        described[slot] = match d {
                            Value::Null => None,
                            Value::String(s) => Some(s.clone()),
                            _ => return Err(bad("every 'criteria' description must be a string")),
                        };
                    }
                }
                let options = ["true", "false"]
                    .iter()
                    .zip(&described)
                    .map(|(n, d)| option_text(n, d.as_deref()))
                    .collect();
                (QType::Noul, options, vec![1, 0])
            }
            Some("score") => {
                let (names, options): (Vec<String>, Vec<String>) = match crit {
                    Some(Value::Array(levels)) => {
                        let mut names = Vec::with_capacity(levels.len());
                        for level in levels {
                            match level {
                                Value::String(s) if !strip(s).is_empty() => {
                                    names.push(strip(s).to_owned())
                                }
                                _ => {
                                    return Err(bad(
                                        "every 'criteria' level must be a non-empty string",
                                    ));
                                }
                            }
                        }
                        (names.clone(), names)
                    }
                    Some(Value::Object(m)) => pairs(m)?
                        .into_iter()
                        .map(|(n, d)| (n.clone(), option_text(&n, d.as_deref())))
                        .unzip(),
                    _ => {
                        return Err(bad(
                            "a score question needs 'criteria', a list or an object of levels",
                        ));
                    }
                };
                if !(MIN_SCORE_LEVELS..=MAX_SCORE_LEVELS).contains(&names.len()) {
                    return Err(bad(&format!(
                        "a score needs {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels, got {}",
                        names.len()
                    )));
                }
                let mut sorted = names.clone();
                sorted.sort();
                sorted.dedup();
                if sorted.len() != names.len() {
                    return Err(bad("'criteria' repeats a level name"));
                }
                let n = options.len();
                (QType::Score, options, (0..n).collect())
            }
            Some("choice") => {
                let pairs = match crit {
                    Some(Value::Object(m)) => pairs(m)?,
                    // A list of labels: {label: null}, each label once.
                    Some(Value::Array(items)) => {
                        let mut m = serde_json::Map::new();
                        for item in items {
                            let k = item
                                .as_str()
                                .ok_or_else(|| bad("choice labels must be strings"))?;
                            m.entry(k.to_owned()).or_insert(Value::Null);
                        }
                        pairs(&m)?
                    }
                    _ => return Err(bad("a choice question needs a non-empty 'criteria' object")),
                };
                if pairs.len() < 2 {
                    return Err(bad("a choice needs at least 2 options"));
                }
                let options: Vec<String> = pairs
                    .iter()
                    .map(|(n, d)| option_text(n, d.as_deref()))
                    .collect();
                let n = options.len();
                (QType::Choice, options, (0..n).collect())
            }
            _ => return Err(bad("'type' must be one of choice, noul, score")),
        };
        let n = options.len();
        let (letters, table) = if n <= self.labels.ids.len() {
            (None, &self.labels)
        } else {
            (
                Some(&self.extended_labels.strings[..]),
                &self.extended_labels,
            )
        };
        let max = self
            .labels
            .ids
            .len()
            .max(self.extended_labels.ids.len().min(MAX_OPTIONS));
        if n > max || n > table.ids.len() {
            return Err(Error::TooManyOptions {
                question: qid.to_owned(),
                options: n,
                head_max_len: max,
            });
        }
        Ok(QuestionPrompt {
            qtype,
            user: user_message(state, &question, &options, letters.map(|l| &l[..n])),
            label_ids: table.ids[..n].to_vec(),
            wire_order,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn config() -> JebadiahConfig {
        JebadiahConfig {
            layout: LAYOUT.into(),
            labels: crate::llm_logits::LabelTable {
                strings: (0..68).map(option_label).collect(),
                ids: (100..168).collect(),
            },
            extended_labels: crate::llm_logits::LabelTable {
                strings: ('A'..='Z')
                    .map(String::from)
                    .chain(('A'..='Z').flat_map(|a| ('A'..='Z').map(move |b| format!("{a}{b}"))))
                    .take(100)
                    .collect(),
                ids: (500..600).collect(),
            },
            max_prompt_tokens: 2048,
        }
    }

    #[test]
    fn labels_are_bijective_base_26() {
        assert_eq!(option_label(0), "A");
        assert_eq!(option_label(25), "Z");
        assert_eq!(option_label(26), "AA");
        assert_eq!(option_label(67), "BP");
        assert_eq!(option_label(254), "IU");
    }

    #[test]
    fn renders_like_ainode() {
        let c = config();
        let state = json!({"z": 1.5, "a": "Zoë"});
        let qs = c
            .questions(
                &state,
                &json!({
                    "n": {"type": "noul", "instructions": "  Refund? ", "criteria": {"false": "no\nrefund  asked"}},
                    "c": {"type": "choice", "instructions": "Team?", "criteria": {" billing ": "cards", "other": null}},
                    "s": {"type": "score", "instructions": "How bad?", "criteria": ["fine", " bad "]},
                }),
            )
            .unwrap();
        assert_eq!(
            qs[0].1.user,
            "STATE:\n{\"a\":\"Zoë\",\"z\":1.5}\n\nQUESTION: Refund?\n\nOPTIONS:\nA. true\n\
             B. false: no refund asked\n\nAnswer with the label of one option and nothing else."
        );
        assert_eq!(qs[0].1.wire_order, [1, 0]);
        assert_eq!(qs[0].1.label_ids, [100, 101]);
        assert!(
            qs[1]
                .1
                .user
                .contains("OPTIONS:\nA. billing: cards\nB. other\n\n")
        );
        assert!(qs[2].1.user.contains("OPTIONS:\nA. fine\nB. bad\n\n"));
        assert_eq!(serialize_state(&json!("plain")), "plain");
        assert_eq!(serialize_state(&Value::Null), "");
    }

    #[test]
    fn rejects_what_ainode_rejects() {
        let c = config();
        for q in [
            json!({"type": "noul", "instructions": {"x": 1}}),
            json!({"type": "noul", "instructions": " "}),
            json!({"type": "noul", "instructions": "x", "criteria": {"yes": "y"}}),
            json!({"type": "choice", "instructions": "x", "criteria": {"a": null}}),
            json!({"type": "choice", "instructions": "x", "criteria": {"a": 1, "b": null}}),
            json!({"type": "score", "instructions": "x", "criteria": ["a"]}),
            json!({"type": "score", "instructions": "x", "criteria": ["a", "a"]}),
            json!({"type": "score", "instructions": "x", "criteria": ["a", 2]}),
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
        let many: Vec<String> = (0..101).map(|i| format!("o{i}")).collect();
        assert!(matches!(
            c.questions(
                &json!("s"),
                &json!({"q": {"type": "choice", "instructions": "x", "criteria": many}})
            ),
            Err(Error::TooManyOptions { options: 101, .. })
        ));
        // Wider than the ainode labels: the extended alphabet, in the text and the candidates.
        let wide: Vec<String> = (0..69).map(|i| format!("o{i}")).collect();
        let q = c
            .questions(
                &json!("s"),
                &json!({"q": {"type": "choice", "instructions": "x", "criteria": wide}}),
            )
            .unwrap();
        assert!(q[0].1.user.contains("\nZ. o25\nAA. o26\n"));
        assert_eq!(q[0].1.label_ids[..2], [500, 501]);
    }
}
