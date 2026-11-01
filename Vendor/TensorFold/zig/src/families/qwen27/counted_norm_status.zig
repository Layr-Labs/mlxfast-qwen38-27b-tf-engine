//! Sticky completion failures read after GPU completion, before token or state acceptance.
pub fn check(failed_word: u32) !void {
    if (failed_word != 0) return error.QwenNormDependencyIncomplete;
}

/// Status records have four u32 words; only word two is a failure flag. Count and arrival words may still be nonzero after an aborted phase.
pub fn checkBank(words: []const u32) !void {
    if (words.len == 0 or words.len % 4 != 0) return error.BadQwenCountedStatus;
    var at: usize = 2;
    while (at < words.len) : (at += 4) try check(words[at]);
}
