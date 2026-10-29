const std = @import("std");

pub fn steps(b: *std.Build, bench_session: *std.Build.Module) void {
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true });
    const engine_api = b.createModule(.{
        .root_source_file = b.path("zig/src/core/engine_api.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{.{ .name = "lanes", .module = lanes }},
    });
    const process_memory = b.createModule(.{ .root_source_file = b.path("zig/src/server/process_memory.zig"), .target = b.graph.host, .optimize = .Debug, .link_libc = true });
    const adapter = b.createModule(.{
        .root_source_file = b.path("zig/src/core/bench_session_engine.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{
            .{ .name = "bench_session", .module = bench_session },
            .{ .name = "engine_api", .module = engine_api },
            .{ .name = "process_memory", .module = process_memory },
        },
    });
    const root = b.createModule(.{
        .root_source_file = b.path("zig/src/core/bench_session_test.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{
            .{ .name = "bench_session", .module = bench_session },
            .{ .name = "bench_session_engine", .module = adapter },
            .{ .name = "engine_api", .module = engine_api },
            .{ .name = "lanes", .module = lanes },
            .{ .name = "process_memory", .module = process_memory },
        },
    });
    b.step("test-bench-session", "Bench session host tests").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = root })).step);
}
