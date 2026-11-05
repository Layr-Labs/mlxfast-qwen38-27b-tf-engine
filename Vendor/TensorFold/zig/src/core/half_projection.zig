//! Non-owning GPU references for exact-half affine projection sidecars.
//! The weight owner retains and releases each referenced Metal buffer.
const mtl = @import("metal");

pub const Ref = struct { buffer: mtl.Buffer, offset: usize = 0 };

/// Exact FP16 high coefficients plus sparse FP32 corrections for a row-major matrix.
/// `columns` is u16 and `offsets` is u32 to match affine_half_sparse.Matrix.
pub const Sidecar = struct {
    high: Ref,
    offsets: Ref,
    columns: Ref,
    deltas: Ref,
    n: u32,
    k: u32,
    corrections: u32,
};
