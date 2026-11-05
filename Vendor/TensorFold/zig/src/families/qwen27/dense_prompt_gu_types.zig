//! Immutable derived coefficients; ownership stays with Weights.owned.
const mtl = @import("metal");

pub const Weights = struct {
    high: mtl.Buffer,
    offsets: mtl.Buffer,
    indices: mtl.Buffer,
    residual: mtl.Buffer,
    count: u32,
    n: u32,
    k: u32,
    layer: u32,
};
