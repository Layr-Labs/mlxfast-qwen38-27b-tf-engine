# Golden status

Official scoring is enabled.

The authoritative pins are in
`fixtures/qwen3_8_27b_tf_mlx_v1_track.json`. The fixture contains:

1. Two timed-pool golden pins.
2. One warm-up golden pin.
3. Two DFlash depth `1` golden pins.
4. One hidden correctness pin.
5. The baseline reference commit.

The ranked preflight verifies every staged file against its byte count and
SHA-256 digest before model work starts.
