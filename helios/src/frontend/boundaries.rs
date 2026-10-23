use tokenizers::{Tokenizer, processors::PostProcessorWrapper};

use super::deepseek::invalid;
use crate::Result;

pub(super) fn preserve_offsets(tokenizer: &mut Tokenizer) {
    fn untrim(processor: &mut PostProcessorWrapper) {
        match processor {
            PostProcessorWrapper::ByteLevel(p) => p.trim_offsets = false,
            PostProcessorWrapper::Roberta(p) => p.trim_offsets = false,
            PostProcessorWrapper::Sequence(p) => p.as_mut().iter_mut().for_each(untrim),
            _ => {}
        }
    }
    if let Some(mut processor) = tokenizer.get_post_processor().cloned() {
        untrim(&mut processor);
        tokenizer.with_post_processor(Some(processor));
    }
}

pub(super) fn token_ends(
    bytes: &[usize],
    offsets: &[(usize, usize)],
    text_len: usize,
) -> Result<Vec<usize>> {
    if bytes.windows(2).any(|w| w[0] > w[1]) {
        return Err(invalid("chat template marks must be in output order"));
    }
    if offsets
        .iter()
        .any(|&(start, end)| start > end || end > text_len)
    {
        return Err(invalid(
            "tokenizer returned offsets outside the rendered prompt",
        ));
    }
    let mut suffix_start = vec![text_len; offsets.len() + 1];
    for i in (0..offsets.len()).rev() {
        suffix_start[i] = suffix_start[i + 1].min(offsets[i].0);
    }
    let mut result = Vec::with_capacity(bytes.len());
    let mut tokens = 0;
    let mut prefix_end = 0;
    for &byte in bytes {
        let byte = byte.min(text_len);
        while tokens < offsets.len() && prefix_end.max(offsets[tokens].1) <= byte {
            prefix_end = prefix_end.max(offsets[tokens].1);
            tokens += 1;
        }
        // Overlapping offsets can represent several tokens for one Unicode scalar.
        if tokens > 0
            && tokens < offsets.len()
            && suffix_start[tokens] >= byte
            && result.last() != Some(&tokens)
        {
            result.push(tokens);
        }
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cuts_require_all_tokens_on_their_respective_side() {
        assert_eq!(
            token_ends(&[0, 1, 2, 3, 4, 6], &[(0, 2), (2, 4), (2, 4), (4, 6)], 6).unwrap(),
            vec![1, 3]
        );
        assert_eq!(
            token_ends(&[2, 3, 4, 5], &[(0, 2), (4, 5), (2, 4), (5, 6)], 6).unwrap(),
            vec![1, 3]
        );
        assert_eq!(
            token_ends(&[2, 3, 4, 100], &[(0, 2), (4, 6)], 6).unwrap(),
            vec![1]
        );
        assert!(token_ends(&[4, 2], &[(0, 2), (2, 6)], 6).is_err());
        assert!(token_ends(&[2], &[(0, 7)], 6).is_err());
    }
}
