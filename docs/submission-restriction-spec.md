# Submission restriction

## Result

A submission can edit only `Vendor/TensorFold/` and `mtp-head.manifest.json`.

1. benchmark.json owns the editable paths.
2. The base commit owns the rule during diff review.
3. A submission cannot edit benchd, tools, workflows, fixtures, or benchmark.json.
4. Symbolic links are refused.
5. A file cannot exceed 2,500,000 bytes.
6. The editable surface cannot exceed 55,829,628 uncompressed bytes.
7. Changed, non-exempt editable paths cannot grow by more than 8,388,608 bytes
   against the review base.
8. The overlay copies only approved paths.
9. The ranked build runs without network access.
10. Evaluator paths stay outside the build read surface.
11. The legacy declaration filename is mtp-head.manifest.json.
12. The declaration is optional.
13. Its default declares serial mode.
14. Edit it to declare DFlash depth `1`.
15. The compressed editable-surface archive cannot exceed 20,971,520 bytes.

The shell tests use temporary repositories and mock workers.
