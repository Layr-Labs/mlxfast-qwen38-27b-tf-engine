//! Hardware policy for the supplied-sums whole-frame prompt schedule.
const std = @import("std");

/// Device names come from the Metal device, not prompts, phases or expected output.
/// Unknown/non-Ultra devices keep 128-row prompt launches.
pub fn wideForDeviceName(name: []const u8) bool {
    return std.mem.startsWith(u8, name, "Apple ") and std.mem.indexOf(u8, name, " Ultra") != null;
}

/// Counted norm is retained only for the validated M5 Max with tensor units.
/// Unknown, Pro and Ultra devices retain ordinary producer/consumer barriers.
pub fn countedForDeviceName(name: []const u8, tensor_units: bool) bool {
    return tensor_units and std.mem.eql(u8, name, "Apple M5 Max");
}

/// A full-frame launch is valid only for a reg prompt with externally supplied sums.
pub fn mayWiden(device_wide: bool, prompt: bool, reg: bool, supplied_sums: bool) bool {
    return device_wide and prompt and reg and supplied_sums;
}

test "device class defaults to conservative launch size" {
    try std.testing.expect(wideForDeviceName("Apple M5 Ultra"));
    try std.testing.expect(wideForDeviceName("Apple M3 Ultra"));
    try std.testing.expect(!wideForDeviceName("Apple M5 Max"));
    try std.testing.expect(!wideForDeviceName("Apple M5 Pro"));
    try std.testing.expect(!wideForDeviceName("Apple M5"));
    try std.testing.expect(!wideForDeviceName(""));
    try std.testing.expect(!wideForDeviceName("Example Ultra"));
}

test "only supplied-sums Ultra reg prompt may widen" {
    inline for (.{ false, true }) |wide| inline for (.{ false, true }) |prompt| inline for (.{ false, true }) |reg| inline for (.{ false, true }) |sums| {
        try std.testing.expectEqual(wide and prompt and reg and sums, mayWiden(wide, prompt, reg, sums));
    };
}

test "counted norm stays within exact tensor-unit Apple M5 Max" {
    try std.testing.expect(countedForDeviceName("Apple M5 Max", true));
    try std.testing.expect(!countedForDeviceName("Apple M5 Max", false));
    try std.testing.expect(!countedForDeviceName("Apple M4 Max", true));
    try std.testing.expect(!countedForDeviceName("Apple M6 Max", true));
    try std.testing.expect(!countedForDeviceName("Apple M5 Max GPU", true));
    try std.testing.expect(!countedForDeviceName("Apple M5 Ultra", true));
    try std.testing.expect(!countedForDeviceName("Apple M5 Pro", true));
    try std.testing.expect(!countedForDeviceName("Apple M5", true));
    try std.testing.expect(!countedForDeviceName("", true));
    try std.testing.expect(!countedForDeviceName("Example Max", true));
}
