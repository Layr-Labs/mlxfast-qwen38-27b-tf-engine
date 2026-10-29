const std = @import("std");

pub fn steps(b: *std.Build, bench_session: *std.Build.Module) void {
    const dflash_audit = b.createModule(.{
        .root_source_file = b.path("zig/src/families/qwen27/dflash/audit.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
    });
    const root = b.createModule(.{
        .root_source_file = b.path("zig/src/families/qwen27/dflash/audit_test.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
        .imports = &.{
            .{ .name = "bench_session", .module = bench_session },
            .{ .name = "dflash_audit", .module = dflash_audit },
        },
    });
    b.step("test-dflash-audit", "DFlash audit host tests").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = root })).step);
}
