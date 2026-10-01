//! `jeeves-markers-v1`: PostHog's Jeeves (fused Qwen3.5-9B with a pointer head) in its no-thinking
//! mode, whose rows follow the upstream `Encoder` (`markers-v3-plainchains`) and `inference.api`
//! (`ollaya_convert.families.jeeves.layout`).
//!
//! ```text
//! content = STATE + esc(render(state)) + "\n" + Q + esc(render(instructions)) + "\n" + block
//! block   = (OPT + esc(option) + OPT_END + "\n")*
//! ids     = tok(pre + strip(content) + post) ⧺ tok("\n") ⧺ tok("</think>\n\n" + block + DECIDE)
//! ```
//!
//! `pre` and `post` are Qwen's chat template around one user message with thinking on (it trims the
//! message, hence `strip`). Every piece is tokenized whole with special tokens parsed, as upstream;
//! `esc` rewrites `<|name|>` to `<¦name¦>`, so user text cannot form a marker. The graph scores
//! option j at the j-th `OPT_END` after the last `</think>` against the decide marker (the last
//! token), as Kev's pointer head does. `render` and the option texts are Kev's.

use serde::Deserialize;
use serde_json::Value;

use crate::Error;
use crate::kev::{escape_delimiters as esc, option_text, render};
use crate::layout::TokenEncoder;
use crate::question::QType;

pub const LAYOUT: &str = "jeeves-markers-v1";
const MAX_OPTIONS: usize = 255;

#[derive(Debug, Clone, Deserialize)]
pub struct Template {
    pub pre: String,
    pub post: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct Markers {
    pub state: String,
    pub question: String,
    pub option_open: String,
    pub option_close: String,
    pub decide: String,
    pub think_end: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MarkerIds {
    pub option_close: u32,
    pub think_end: u32,
}

/// The layout as `decision.json` declares it.
#[derive(Debug, Clone, Deserialize)]
pub struct JeevesLayout {
    pub template: Template,
    pub markers: Markers,
    pub marker_ids: MarkerIds,
    pub empty_think: String,
    pub max_row_tokens: usize,
    pub pad: u32,
}

/// One question's row: token ids, the decide position and the option positions.
#[derive(Debug, Clone, PartialEq)]
pub struct JeevesRow {
    pub ids: Vec<u32>,
    pub decide: usize,
    pub opts: Vec<usize>,
    pub qtype: QType,
}

fn py_isspace(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

impl JeevesLayout {
    pub fn validate(&self) -> Result<(), Error> {
        if self.max_row_tokens == 0 || self.template.post.is_empty() {
            return Err(Error::invalid(
                "decision.json: max_row_tokens and the template are required",
            ));
        }
        Ok(())
    }

    /// `inference.api.parse_question` + `Question.options`: the question's type and option texts.
    fn options(&self, qid: &str, def: &Value) -> Result<(QType, Vec<String>), Error> {
        let bad = |msg: &str| Error::invalid(format!("question {qid:?}: {msg}"));
        let q = def.as_object().ok_or_else(|| bad("must be an object"))?;
        if q.keys()
            .any(|k| !matches!(k.as_str(), "type" | "instructions" | "criteria"))
        {
            return Err(bad(
                "unknown fields; a question takes type, instructions and criteria",
            ));
        }
        let crit = q.get("criteria").unwrap_or(&Value::Null);
        match q.get("type").and_then(Value::as_str) {
            Some("choice") => {
                let m = crit
                    .as_object()
                    .filter(|m| (1..=MAX_OPTIONS).contains(&m.len()))
                    .ok_or_else(|| {
                        bad("this model takes choice criteria as an object of 1..255 options")
                    })?;
                Ok((
                    QType::Choice,
                    m.iter().map(|(k, v)| option_text(k, Some(v))).collect(),
                ))
            }
            Some("noul") => {
                let empty = serde_json::Map::new();
                let m = match crit {
                    Value::Null => &empty,
                    Value::Object(m) if m.keys().all(|k| k == "true" || k == "false") => m,
                    _ => return Err(bad("noul criteria may only describe true and false")),
                };
                Ok((
                    QType::Noul,
                    vec![
                        option_text("no", m.get("false")),
                        option_text("yes", m.get("true")),
                    ],
                ))
            }
            Some("score") => {
                let levels = crit
                    .as_array()
                    .filter(|l| (1..=MAX_OPTIONS).contains(&l.len()))
                    .ok_or_else(|| bad("score criteria must be a list of 1..255 levels"))?;
                Ok((QType::Score, levels.iter().map(render).collect()))
            }
            _ => Err(bad("type must be one of choice, noul, score")),
        }
    }

    /// Every question's row, in request order.
    pub fn rows(
        &self,
        tok: &dyn TokenEncoder,
        state: &Value,
        defs: &[(&str, &Value)],
    ) -> Result<Vec<JeevesRow>, Error> {
        if defs.is_empty() {
            return Err(Error::invalid("questions must be a non-empty object"));
        }
        let m = &self.markers;
        let state_text = render(state);
        let state_text = esc(&state_text);
        defs.iter()
            .map(|(qid, def)| {
                let (qtype, options) = self.options(qid, def)?;
                let mut block = String::new();
                for o in &options {
                    block.push_str(&m.option_open);
                    block.push_str(&esc(o));
                    block.push_str(&m.option_close);
                    block.push('\n');
                }
                let instructions = render(def.get("instructions").unwrap_or(&Value::Null));
                let content = format!(
                    "{}{state_text}\n{}{}\n{block}",
                    m.state,
                    m.question,
                    esc(&instructions)
                );
                let prompt = format!(
                    "{}{}{}",
                    self.template.pre,
                    content.trim_matches(py_isspace),
                    self.template.post
                );
                let mut ids = tok.encode(&prompt)?;
                ids.extend(tok.encode(&self.empty_think)?);
                ids.extend(tok.encode(&format!("{}\n\n{block}{}", m.think_end, m.decide))?);
                if ids.len() > self.max_row_tokens {
                    return Err(Error::invalid(format!(
                        "question {qid:?}: the row is {} tokens; this model reads up to {}",
                        ids.len(),
                        self.max_row_tokens
                    )));
                }
                let start = ids
                    .iter()
                    .rposition(|&t| t == self.marker_ids.think_end)
                    .unwrap_or(0);
                let opts: Vec<usize> = (start..ids.len())
                    .filter(|&i| ids[i] == self.marker_ids.option_close)
                    .collect();
                if opts.len() != options.len() {
                    return Err(Error::invalid(format!(
                        "question {qid:?}: {} option markers for {} options",
                        opts.len(),
                        options.len()
                    )));
                }
                Ok(JeevesRow {
                    decide: ids.len() - 1,
                    ids,
                    opts,
                    qtype,
                })
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// One token per char, and the markers (as specials) one token each.
    struct Tok;
    impl TokenEncoder for Tok {
        fn encode(&self, text: &str) -> Result<Vec<u32>, Error> {
            let mut out = Vec::new();
            let mut rest = text;
            'outer: while let Some(c) = rest.chars().next() {
                for (m, id) in [("<|box_end|>", 1_000_001), ("</think>", 1_000_002)] {
                    if let Some(r) = rest.strip_prefix(m) {
                        out.push(id);
                        rest = r;
                        continue 'outer;
                    }
                }
                out.push(c as u32);
                rest = &rest[c.len_utf8()..];
            }
            Ok(out)
        }
    }

    fn layout() -> JeevesLayout {
        JeevesLayout {
            template: Template {
                pre: "<|im_start|>user\n".into(),
                post: "<|im_end|>\n<|im_start|>assistant\n<think>\n".into(),
            },
            markers: Markers {
                state: "<|fim_prefix|>".into(),
                question: "<|fim_middle|>".into(),
                option_open: "<|box_start|>".into(),
                option_close: "<|box_end|>".into(),
                decide: "<|fim_suffix|>".into(),
                think_end: "</think>".into(),
            },
            marker_ids: MarkerIds {
                option_close: 1_000_001,
                think_end: 1_000_002,
            },
            empty_think: "\n".into(),
            max_row_tokens: 8192,
            pad: 0,
        }
    }

    #[test]
    fn reads_the_repeated_option_block() {
        let q = json!({"type": "noul", "instructions": "Refund <|box_end|>?", "criteria": {"true": "money back"}});
        let rows = layout().rows(&Tok, &json!({"a": 1}), &[("r", &q)]).unwrap();
        let r = &rows[0];
        assert_eq!(r.opts.len(), 2);
        assert_eq!(r.decide, r.ids.len() - 1);
        // The escaped marker in the question stays text; the first block's markers precede </think>.
        let think = r.ids.iter().rposition(|&t| t == 1_000_002).unwrap();
        assert!(r.opts.iter().all(|&p| p > think));
        assert_eq!(r.ids.iter().filter(|&&t| t == 1_000_001).count(), 4);
    }

    #[test]
    fn rejects_what_upstream_rejects() {
        let l = layout();
        for q in [
            json!({"type": "noul", "instructions": "x", "criteria": {"yes": "y"}}),
            json!({"type": "choice", "instructions": "x", "criteria": ["a", "b"]}),
            json!({"type": "score", "instructions": "x", "criteria": {"a": 1}}),
            json!({"type": "noul", "instructions": "x", "extra": 1}),
        ] {
            assert!(l.rows(&Tok, &json!("s"), &[("q", &q)]).is_err(), "{q}");
        }
    }
}
