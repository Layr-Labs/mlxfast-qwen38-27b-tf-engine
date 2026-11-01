//! Sticky completion failure read only after GPU completion, before token selection.
pub fn check(failed_word: u32) !void {
    if (failed_word != 0) return error.QwenNormDependencyIncomplete;
}
