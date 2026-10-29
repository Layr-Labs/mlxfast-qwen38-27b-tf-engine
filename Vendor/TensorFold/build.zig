// Modified by Layr Labs, Inc. in 2026; see THIRD_PARTY_NOTICES.md.
const metal_build = @import("zig/build/metal_targets.zig");
const std = @import("std");
const glm_expiry_build = @import("zig/build/glm_expiry.zig");
const builtin = @import("builtin");
const cuda_build = @import("zig/build/cuda.zig");
const dist_build = @import("zig/build/dist.zig");
const qwen27_build = @import("zig/build/qwen27.zig");
const core_checks = @import("zig/build/core_checks.zig");
const glm_checks = @import("zig/build/glm_checks.zig");
const merge_build = @import("zig/build/merge.zig");
const flashnext_checks = @import("zig/build/flashnext_checks.zig");
const hip_build = @import("zig/build/hip.zig");
const xpu_build = @import("zig/build/xpu.zig");
const grammar_build = @import("zig/build/grammar.zig");
const counters_build = @import("zig/build/metal_counters.zig");
const exl3_build = @import("zig/build/exl3.zig");
const exlr8_build = @import("zig/build/exlr8.zig");
const server_build = @import("zig/build/server.zig");
const golden_build = @import("zig/build/golden.zig");
const contract_build = @import("zig/build/kernel_contract.zig");
const video_build = @import("zig/build/video.zig");
const sys_build = @import("zig/build/sys.zig");

comptime {
    const required = std.mem.trim(u8, @embedFile(".zig-version"), "\r\n");
    if (!std.mem.eql(u8, builtin.zig_version_string, required))
        @compileError("TensorFold's Zig engine requires Zig " ++ required);
}

const metal_lists = @import("zig/build/metal_source_lists.zig");
const kernels = metal_lists.kernels;
const generated = metal_lists.generated;
const kimi_dir = metal_lists.kimi_dir;
const kimi_kernels = metal_lists.kimi_kernels;
const nemotron_packed = metal_lists.nemotron_packed;

/// MLX's language version for mx.fast.metal_kernel on this macOS: 4.1 from 27, 4.0 on 26, 3.2 on 15, else 3.1.
const mlxMetalStd = metal_build.mlxMetalStd;

/// xcrun metal -c each source to .air with `flags`, then link them into `name`.metallib.
const metallib = metal_build.metallib;

/// The Kimi metallib: each source compiled with kimi_dir on the include path, the shared header a dependency.
const kimiMetallib = metal_build.kimiMetallib;

/// A module whose `bytes` is the metallib at `lib`, embedded at compile time.
const embedded = metal_build.embedded;

/// The page pool's own tests, and its checks under engine_api's paged store and lane host (kv_pool a module of its...
fn poolChecks(b: *std.Build, target: std.Build.ResolvedTarget, test_step: *std.Build.Step) void {
    const pool = b.createModule(.{ .root_source_file = b.path("zig/src/core/kv_pool.zig"), .target = target, .optimize = .Debug });
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = .Debug, .link_libc = true });
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = .Debug, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const checks = b.createModule(.{ .root_source_file = b.path("zig/src/core/kv_pool_checks.zig"), .target = target, .optimize = .Debug, .link_libc = true, .imports = &.{ .{ .name = "kv_pool", .module = pool }, .{ .name = "engine_api", .module = api }, .{ .name = "lanes", .module = lanes } } });
    const step = b.step("test-kv-pool", "The KV page pool: layout, leases, injected failures, the paged store and lane-host admission");
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = pool })).step);
    step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = checks })).step);
    test_step.dependOn(step);
}

fn benchChecks(b: *std.Build) void {
    const bench_session = b.createModule(.{ .root_source_file = b.path("zig/src/core/bench_session.zig"), .target = b.graph.host, .optimize = .Debug });
    const protocol = b.createModule(.{
        .root_source_file = b.path("zig/src/bench/worker_test.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
        .imports = &.{.{ .name = "bench_session", .module = bench_session }},
    });
    b.step("test-bench-protocol", "Engine Protocol v1 worker host tests").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = protocol })).step);
    const resident = b.createModule(.{
        .root_source_file = b.path("zig/src/bench/resident_test.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
        .link_libc = true,
        .imports = &.{.{ .name = "bench_session", .module = bench_session }},
    });
    b.step("test-bench-resident", "Bench resident Unix socket tests").dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = resident })).step);
    @import("zig/build/bench_session_host.zig").steps(b, bench_session);
    @import("zig/build/dflash_audit_host.zig").steps(b, bench_session);
}

pub fn build(b: *std.Build) void {
    @import("zig/build/residency_host.zig").steps(b);
    @import("zig/build/lane_halt_host.zig").steps(b);
    @import("zig/build/dsv41_shared.zig").steps(b);
    @import("zig/build/dsv41_cuda.zig").steps(b);
    @import("zig/build/dsv41_state.zig").steps(b);
    @import("zig/build/dsv41_draft.zig").steps(b);
    const target = b.standardTargetOptions(.{});
    benchChecks(b);
    @import("zig/build/copy_source.zig").steps(b);
    const release_version = b.option([]const u8, "version", "Release version, exactly MAJOR.MINOR.PATCH");
    if (release_version) |v| if (!dist_build.validVersion(v)) @panic("-Dversion must be exactly MAJOR.MINOR.PATCH");
    const build_options = b.addOptions();
    // Keep the existing manifest version for developer builds; releases require an explicit override.
    const manifest = @import("build.zig.zon");
    build_options.addOption([]const u8, "version", release_version orelse manifest.version);
    const slide_test_hooks = b.option(bool, "slide-test-hooks", "Living Weights test hooks, such as TF_SLIDE_FORGET_FAIL (never in a release)") orelse false;
    if (slide_test_hooks and release_version != null) @panic("-Dslide-test-hooks is for test builds: a -Dversion release never carries it");
    build_options.addOption(bool, "slide_test_hooks", slide_test_hooks);
    const optimize = b.option(std.builtin.OptimizeMode, "optimize", "debug, safe, fast or small (default fast)") orelse .fast;
    // Nemotron's draft vocabulary (zig/src/families/nemotron/draft_ids.txt), for every backend
    const ids_files = b.addWriteFiles();
    _ = ids_files.addCopyFile(b.path("zig/src/families/nemotron/draft_ids.txt"), "draft_ids.txt");
    const draft_ids = b.createModule(.{ .root_source_file = ids_files.add("draft_ids.zig", "pub const text = @embedFile(\"draft_ids.txt\");\n") });
    const test_step = b.step("test", "Unit tests (macOS includes available Metal device checks)");
    @import("zig/build/kernel_symbols.zig").tests(b, test_step);
    @import("zig/build/checkpoint_security.zig").tests(b, test_step);
    if (b.graph.host.result.os.tag == .macos) {
        const step5_mods = metalEngineModules(b, b.graph.host, .Debug, draft_ids);
        const empty = b.addWriteFiles().add("step5-compile-only.metallib", "");
        step5_mods.sources.addImport("nemotron_packed_metallib", embedded(b, empty, "nemotron_packed.metallib"));
        @import("zig/build/step5_host.zig").steps(b, step5_mods.metal, step5_mods.engine, step5_mods.lanes);
    }
    @import("zig/build/h3_host.zig").tests(b, target, test_step);
    @import("zig/build/digest.zig").steps(b, test_step, draft_ids, build_options);
    @import("zig/build/flashnext_budget.zig").steps(b, test_step);
    @import("zig/build/flashnext_batch_host.zig").tests(b, test_step);
    if (b.graph.host.result.os.tag == .macos) {
        const budget_mods = metalEngineModules(b, b.graph.host, .Debug, draft_ids);
        const budget_empty = b.addWriteFiles().add("budget-compile-only.metallib", "");
        budget_mods.sources.addImport("nemotron_packed_metallib", embedded(b, budget_empty, "nemotron_packed.metallib"));
        @import("zig/build/flashnext_budget.zig").compileHost(b, budget_mods.metal, budget_mods.engine, budget_mods.lanes);
        @import("zig/build/residency_host.zig").compileHost(b, budget_mods.metal, budget_mods.engine, budget_mods.lanes);
    }
    // every tracked path checks out on Windows too
    const names_run = b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/tests/portable_names.zig"), .target = target }) }));
    names_run.setCwd(b.path("."));
    test_step.dependOn(&names_run.step);
    // the XPU server's stream admission, a pure function of the card's memory
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/native/xpu_admit.zig"), .target = target }) })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/safetensors.zig"), .target = target }) })).step);
    test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = b.createModule(.{ .root_source_file = b.path("zig/src/xpu/limits.zig"), .target = target }) })).step);
    const cost_cases = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/cost_rule_cases.zig"),
        .target = target,
        .optimize = .Debug,
        .link_libc = true,
    }) });
    const cost_run = b.addRunArtifact(cost_cases);
    b.step("test-cost-rule", "CPU scripted window and interleaved head timing").dependOn(&cost_run.step);
    test_step.dependOn(&cost_run.step);
    const identity_cases = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/core/lanes/cost_identity.zig"),
        .target = b.graph.host,
        .optimize = .Debug,
        .link_libc = true,
    }) });
    const identity_run = b.addRunArtifact(identity_cases);
    b.step("test-cost-identity", "Streamed window-cost identity matches legacy bytes with bounded memory").dependOn(&identity_run.step);
    test_step.dependOn(&identity_run.step);
    merge_build.targets(b, target, optimize); // tensorfold merge's tests and tf-merge-check, host only on every OS
    const voice_tool = b.createModule(.{ .root_source_file = b.path("zig/src/server/drama_voice_tool.zig"), .target = target, .optimize = optimize });
    _ = server_build.addTo(b, voice_tool, target, optimize);
    const voice_exe = b.addExecutable(.{ .name = "drama3-voice", .root_module = voice_tool });
    b.step("drama3-voice", "Attested Drama voice files from reference WAVs, coded by drama3-generate --encode").dependOn(&b.addInstallArtifact(voice_exe, .{}).step);
    // the backend options are declared for every target, so one build script may pass the same flags on every OS
    const cuda_options = @import("zig/build/options.zig").cuda(b);
    const tp_options = @import("zig/build/options.zig").tp(b);
    switch (target.result.os.tag) {
        .macos => {
            metalTargets(b, target, optimize, draft_ids, build_options, test_step);
            benchWorkerMetal(b, target, optimize, draft_ids, build_options);
        },
        .linux, .windows => {
            cuda_build.targets(b, target, optimize, draft_ids, build_options, cuda_options);
            benchWorkerCuda(b, target, optimize, draft_ids, build_options, cuda_options);
        },
        else => {},
    }
    const gemma4_host_sources = b.createModule(.{ .root_source_file = b.path("zig/kernels/metal/sources.zig"), .target = b.graph.host });
    gemma4_host_sources.addAnonymousImport("nemotron_packed_metallib", .{ .root_source_file = b.addWriteFiles().add("gemma4-host-empty.metallib", "") });
    const gemma4_contract_host = @import("zig/build/gemma4_host.zig").tests(b, gemma4_host_sources);
    gemma4_contract_host.dependOn(@import("zig/build/glm_rows_contract.zig").host(b, gemma4_host_sources));
    if (b.graph.host.result.os.tag == .macos) {
        const gemma4_host_mods = metalEngineModules(b, b.graph.host, .Debug, draft_ids);
        gemma4_host_mods.sources.addImport("nemotron_packed_metallib", embedded(b, b.addWriteFiles().add("gemma4-compile-empty.metallib", ""), "nemotron_packed.metallib"));
        @import("zig/build/gemma4_host.zig").compileHost(b, gemma4_host_mods.metal, gemma4_host_mods.engine, gemma4_host_mods.lanes, gemma4_host_mods.sources);
        @import("zig/build/qwen27_host.zig").compileHost(b, gemma4_host_mods.metal, gemma4_host_mods.engine, gemma4_host_mods.lanes);
    }
    @import("zig/build/qwen27_host.zig").tests(b);
    const xgrammar = grammar_build.option(b);
    dist_build.targets(b, draft_ids, build_options, release_version, xgrammar);
    @import("zig/build/sessions.zig").add(b, test_step);
    @import("zig/build/tp.zig").add(b, target, optimize, test_step, tp_options);
    @import("zig/build/gradient_host.zig").add(b);
    @import("zig/build/living_tools_host.zig").steps(b);
    cuda_build.hostTests(b, draft_ids, build_options, test_step, target.result.os.tag != .macos);
    xpu_build.hostTests(b, test_step);
    if (target.result.os.tag == .linux or target.result.os.tag == .windows) {
        // the engine seam's tests (lane_host.zig and its siblings), which the macOS branch runs from its own api module
        const lanes_mod = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
        const api_mod = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes_mod }} });
        test_step.dependOn(&b.addRunArtifact(b.addTest(.{ .root_module = api_mod })).step);
    }
    poolChecks(b, b.graph.host, test_step);
    const with_hip = b.option(bool, "hip", "Enable the opt-in HIP backend and its host/device targets") orelse false;
    const hip_options = hip_build.options(b); // declared on every target (#1165), used only with -Dhip=true
    hip_build.loaderTests(b, test_step);
    if (with_hip) hip_build.steps(b, target, optimize, build_options, test_step, hip_options);
    exlr8_build.hostTests(b, target, test_step);
    xpu_build.steps(b, target, optimize, build_options);
    video_build.steps(b, target, test_step);
    const lanes_cpu = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = b.graph.host, .optimize = .debug, .link_libc = true });
    const disk_cpu = b.addTest(.{ .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/glm_disk_tests.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "lanes", .module = lanes_cpu }},
    }) });
    const disk_run = b.addRunArtifact(disk_cpu);
    test_step.dependOn(&disk_run.step);
    b.step("test-learn-recovery", "CPU learned-state recovery and admission tests").dependOn(&disk_run.step);
    @import("zig/build/qwen27_required.zig").tests(b, lanes_cpu);
    glm_expiry_build.steps(b, test_step, lanes_cpu);
    const warning = b.addExecutable(.{ .name = "learned-warning-test", .root_module = b.createModule(.{
        .root_source_file = b.path("zig/src/learned_warning_test.zig"),
        .target = b.graph.host,
        .optimize = .debug,
        .link_libc = true,
        .imports = &.{.{ .name = "lanes", .module = lanes_cpu }},
    }) });
    const warning_run = b.addRunArtifact(warning);
    test_step.dependOn(&warning_run.step);
    b.step("test-learn-warning", "CPU production-log refusal, backoff and recovery test").dependOn(&warning_run.step);
    grammar_build.steps(b, target, xgrammar);
    grammar_build.hostTests(b, test_step);
    contract_build.hostTests(b, gemma4_contract_host);
    // the CPU-only fake server and engine-less CLI (zig build server-cpu), behind zig/tests/server/build.sh
    @import("zig/build/server.zig").cpuArtefacts(b, b.graph.host, optimize, build_options);
    sys_build.wire(b);
    @import("zig/build/checkpoint_config.zig").wire(b);
    @import("zig/build/save_guard.zig").wire(b);
    @import("zig/build/dev_tools.zig").wire(b, target);
    if (b.graph.host.result.os.tag == .macos) routeMetalModuleCache(b);
}

fn routeMetalModuleCache(b: *std.Build) void {
    var seen: std.AutoHashMapUnmanaged(*std.Build.Step, void) = .empty;
    for (b.top_level_steps.values()) |top| routeMetalStep(b, &seen, &top.step);
    for (b.graph.generated_files.items) |step| routeMetalStep(b, &seen, step);
}

fn routeMetalStep(b: *std.Build, seen: *std.AutoHashMapUnmanaged(*std.Build.Step, void), step: *std.Build.Step) void {
    if ((seen.getOrPut(b.allocator, step) catch @panic("OOM")).found_existing) return;
    if (step.cast(std.Build.Step.Run)) |run| {
        const tool = if (run.argv.items.len > 3) switch (run.argv.items[3]) {
            .bytes => |value| value,
            else => "",
        } else "";
        if (std.mem.eql(u8, tool, "metal"))
            run.addArg("-fmodules-cache-path=./.zig-cache/metal-modules");
    }
    for (step.dependencies.items) |dependency| routeMetalStep(b, seen, dependency);
}

/// The native Metal server build is owned by the shared Metal target module.
const nativeServer = metal_build.nativeServer;

/// macOS: the metallib, the Metal runtime's test programs and the engine over Metal.
const metalTargets = metal_build.metalTargets;

/// The server and distribution use the same embedded-source Metal engine.
const metalEngineModules = metal_build.metalEngineModules;

pub const distMetalServer = metal_build.distMetalServer;

fn benchWorkerMetal(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options) void {
    const mods = metalEngineModules(b, target, optimize, draft_ids);
    const flags = [_][]const u8{ b.fmt("-std={s}", .{mlxMetalStd(b)}), "-fmetal-math-mode=safe", "-fmetal-math-fp32-functions=fast", "-fmodules-cache-path=./.zig-cache/metal-modules" };
    mods.sources.addImport("nemotron_packed_metallib", embedded(b, metallib(b, "bench_nemotron_packed", &nemotron_packed, &flags), "nemotron_packed.metallib"));

    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = mods.lanes }} });
    const native = b.createModule(.{
        .root_source_file = b.path("zig/src/native/metal.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "metal", .module = mods.metal }, .{ .name = "engine_api", .module = api }, .{ .name = "tensorfold", .module = mods.engine } },
    });
    addBenchDrama(b, target, optimize, mods.metal, native);

    const session = b.createModule(.{ .root_source_file = b.path("zig/src/core/bench_session.zig"), .target = target, .optimize = optimize });
    const adapter = b.createModule(.{
        .root_source_file = b.path("zig/src/core/bench_session_engine.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "bench_session", .module = session }, .{ .name = "engine_api", .module = api } },
    });
    const server_engines = b.createModule(.{
        .root_source_file = b.path("zig/src/server/engines.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "native_engines", .module = native } },
    });
    server_engines.addOptions("build_options", build_options);
    const root = b.createModule(.{
        .root_source_file = b.path("zig/src/bench/main.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "engine_api", .module = api },
            .{ .name = "bench_session", .module = session },
            .{ .name = "bench_session_engine", .module = adapter },
            .{ .name = "server_engines", .module = server_engines },
            .{ .name = "core", .module = mods.engine },
            .{ .name = "metal", .module = mods.metal },
        },
    });
    const exe = b.addExecutable(.{ .name = "bench-worker", .root_module = root });
    b.step("bench-worker", "Build the resident Engine Protocol v1 worker").dependOn(&b.addInstallArtifact(exe, .{}).step);
}

fn addBenchDrama(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, metal: *std.Build.Module, native: *std.Build.Module) void {
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = .ReleaseSafe, .link_libc = true });
    native.addImport("tokenizer", tokenizer);
    const rounds = b.createModule(.{ .root_source_file = b.path("zig/src/core/frame_rounds.zig"), .target = target, .optimize = optimize });
    const shim = b.addWriteFiles().add("bench_drama3_core.zig", "pub const safetensors = @import(\"tensorfold\").safetensors;\npub const tokenizer = @import(\"tokenizer\");\npub const frame_rounds = @import(\"frame_rounds\");\n");
    const core = b.createModule(.{ .root_source_file = shim, .target = target, .optimize = optimize, .imports = &.{
        .{ .name = "tensorfold", .module = native.import_table.get("tensorfold").? },
        .{ .name = "tokenizer", .module = tokenizer },
        .{ .name = "frame_rounds", .module = rounds },
    } });
    const dir = "zig/kernels/metal/drama3";
    const sources = [_][]const u8{ "dense.metal", "gdn.metal", "attn.metal", "moe.metal", "nvfp4.metal", "guidance.metal", "sample.metal", "codec.metal" };
    const link = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metallib", "-o" });
    const lib = link.addOutputFileArg("bench_drama3.metallib");
    for (sources) |name| {
        const cc = b.addSystemCommand(&.{ "xcrun", "-sdk", "macosx", "metal", b.fmt("-std={s}", .{mlxMetalStd(b)}), "-fmetal-math-mode=safe", "-fmetal-math-fp32-functions=precise", "-fmodules-cache-path=./.zig-cache/metal-modules" });
        cc.addPrefixedDirectoryArg("-I", b.path(dir));
        cc.addFileInput(b.path(dir ++ "/d3_common.h"));
        cc.addArg("-c");
        cc.addFileArg(b.path(b.fmt("{s}/{s}", .{ dir, name })));
        cc.addArg("-o");
        link.addFileArg(cc.addOutputFileArg(b.fmt("bench_d3_{s}.air", .{std.fs.path.stem(name)})));
    }
    const bytes = embedded(b, lib, "drama3.metallib");
    const family = b.createModule(.{
        .root_source_file = b.path("zig/src/families/drama3/drama3_metal.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{ .{ .name = "core", .module = core }, .{ .name = "metal", .module = metal }, .{ .name = "drama3_metallib", .module = bytes } },
    });
    native.addImport("core", core);
    native.addImport("drama3", family);
}

fn benchWorkerCuda(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, draft_ids: *std.Build.Module, build_options: *std.Build.Step.Options, opts: @import("zig/build/options.zig").Cuda) void {
    const version: ?std.Build.LazyPath = if (opts.fatbins == null and opts.nvcc != null) blk: {
        const run = b.addSystemCommand(&.{ opts.nvcc.?, "--version" });
        run.has_side_effects = true;
        break :blk run.captureStdOut(.{});
    } else null;
    var images: [cuda_build.kernels.len]?std.Build.LazyPath = @splat(null);
    for (cuda_build.kernels, &images) |kernel, *image| {
        image.* = if (opts.fatbins) |dir|
            b.graph.cwdRelativePath(b.pathJoin(&.{ dir, b.fmt("{s}.fatbin", .{kernel.name}) }))
        else if (opts.nvcc) |tool|
            cuda_build.fatbin(b, tool, version.?, kernel, opts.sms)
        else
            null;
    }
    const cuda = cuda_build.runtime(b, target, optimize, if (opts.nvcc != null or opts.fatbins != null) &images else &.{});
    const mods = benchCudaFamily(b, target, optimize, cuda, draft_ids);
    const native = benchCudaEngines(b, target, optimize, cuda, mods.core, mods.lanes, mods.nemotron, mods.heat, mods.qwen35, mods.qwen_vision);
    addBenchExecutable(b, target, optimize, build_options, native.api, native.engines, mods.core, null);
}

const BenchCudaFamily = struct {
    core: *std.Build.Module,
    lanes: *std.Build.Module,
    nemotron: *std.Build.Module,
    qwen35: *std.Build.Module,
    qwen_vision: *std.Build.Module,
    heat: *std.Build.Module,
};

fn benchCudaFamily(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, draft_ids: *std.Build.Module) BenchCudaFamily {
    const tokenizer = b.createModule(.{ .root_source_file = b.path("zig/src/core/tokenizer/tokenizer.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const core = b.createModule(.{ .root_source_file = b.path("zig/src/core/root.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("gpu_memory", @import("zig/build/gpu_memory.zig").module(b, target));
    core.addImport("tokenizer", tokenizer);
    const lanes = b.createModule(.{ .root_source_file = b.path("zig/src/core/lanes/lanes.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const heat = b.createModule(.{ .root_source_file = b.path("zig/src/core/heat.zig"), .target = target, .optimize = optimize, .link_libc = true });
    const living = b.createModule(.{ .root_source_file = b.path("zig/src/living/living.zig"), .target = target, .optimize = optimize, .link_libc = true });
    core.addImport("living", living);
    const nemotron = b.createModule(.{ .root_source_file = b.path("zig/src/families/nemotron/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    for ([_]std.Build.Module.Import{
        .{ .name = "living", .module = living },
        .{ .name = "cuda", .module = cuda },
        .{ .name = "core", .module = core },
        .{ .name = "lanes", .module = lanes },
        .{ .name = "heat", .module = heat },
        .{ .name = "nemotron_draft_ids", .module = draft_ids },
    }) |item| nemotron.addImport(item.name, item.module);
    const radio = b.createModule(.{ .root_source_file = b.path("zig/src/families/radio/radio.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes } } });
    nemotron.addImport("radio", radio);
    const qwen35 = b.createModule(.{ .root_source_file = b.path("zig/src/families/qwen3_5/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true });
    for ([_]std.Build.Module.Import{
        .{ .name = "living", .module = living },
        .{ .name = "nemotron", .module = nemotron },
        .{ .name = "cuda", .module = cuda },
        .{ .name = "core", .module = core },
        .{ .name = "lanes", .module = lanes },
    }) |item| qwen35.addImport(item.name, item.module);
    const vision = b.createModule(.{ .root_source_file = b.path("zig/src/vision/qwen_cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes } } });
    qwen35.addImport("qwen_vision", vision);
    return .{ .core = core, .lanes = lanes, .nemotron = nemotron, .qwen35 = qwen35, .qwen_vision = vision, .heat = heat };
}

fn benchCudaEngines(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, cuda: *std.Build.Module, core: *std.Build.Module, lanes: *std.Build.Module, nemotron: *std.Build.Module, heat: *std.Build.Module, qwen35: *std.Build.Module, vision: *std.Build.Module) struct { api: *std.Build.Module, engines: *std.Build.Module } {
    const api = b.createModule(.{ .root_source_file = b.path("zig/src/core/engine_api.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{.{ .name = "lanes", .module = lanes }} });
    const lfm2 = b.createModule(.{ .root_source_file = b.path("zig/src/families/lfm2/lfm2.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core } } });
    const flashnext = b.createModule(.{ .root_source_file = b.path("zig/src/families/flashnext/cuda_launch.zig"), .target = target, .optimize = optimize, .link_libc = true });
    for ([_]std.Build.Module.Import{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "lanes", .module = lanes }, .{ .name = "qwen_vision", .module = vision }, .{ .name = "living", .module = core.import_table.get("living").? } }) |item| flashnext.addImport(item.name, item.module);
    const canon = b.createModule(.{ .root_source_file = b.path("zig/src/cluster/canon.zig"), .target = target, .optimize = optimize });
    const glm = b.createModule(.{ .root_source_file = b.path("zig/src/families/glm/cuda.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "cuda", .module = cuda }, .{ .name = "core", .module = core }, .{ .name = "canon", .module = canon }, .{ .name = "lanes", .module = lanes }, .{ .name = "heat", .module = heat } } });
    const kolibri = @import("zig/build/kolibri.zig").module(b, target, optimize, cuda, core, lanes);
    const engines = b.createModule(.{
        .root_source_file = b.path("zig/src/native/cuda.zig"),
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .imports = &.{
            .{ .name = "cuda", .module = cuda },
            .{ .name = "engine_api", .module = api },
            .{ .name = "lanes", .module = lanes },
            .{ .name = "nemotron", .module = nemotron },
            .{ .name = "drama3", .module = @import("zig/build/drama3_host.zig").module(b, target, optimize, cuda, core) },
            .{ .name = "lfm2", .module = lfm2 },
            .{ .name = "core", .module = core },
            .{ .name = "flashnext", .module = flashnext },
            .{ .name = "glm_cuda", .module = glm },
            .{ .name = "qwen35", .module = qwen35 },
            .{ .name = "kolibri1", .module = kolibri },
            .{ .name = "gemma4_cuda", .module = @import("zig/build/gemma4_cuda.zig").module(b, target, optimize, cuda, core, lanes) },
            .{ .name = "spark25", .module = @import("zig/build/spark25.zig").module(b, target, optimize, cuda, core, lanes) },
            .{ .name = "dory", .module = @import("zig/build/dory.zig").module(b, target, optimize, cuda, core, lanes) },
        },
    });
    return .{ .api = api, .engines = engines };
}

fn addBenchExecutable(b: *std.Build, target: std.Build.ResolvedTarget, optimize: std.builtin.OptimizeMode, build_options: *std.Build.Step.Options, api: *std.Build.Module, native: *std.Build.Module, core: *std.Build.Module, metal: ?*std.Build.Module) void {
    const session = b.createModule(.{ .root_source_file = b.path("zig/src/core/bench_session.zig"), .target = target, .optimize = optimize });
    const adapter = b.createModule(.{ .root_source_file = b.path("zig/src/core/bench_session_engine.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "bench_session", .module = session }, .{ .name = "engine_api", .module = api } } });
    const server_engines = b.createModule(.{ .root_source_file = b.path("zig/src/server/engines.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "native_engines", .module = native } } });
    server_engines.addOptions("build_options", build_options);
    const root = b.createModule(.{ .root_source_file = b.path("zig/src/bench/main.zig"), .target = target, .optimize = optimize, .link_libc = true, .imports = &.{ .{ .name = "engine_api", .module = api }, .{ .name = "bench_session", .module = session }, .{ .name = "bench_session_engine", .module = adapter }, .{ .name = "server_engines", .module = server_engines }, .{ .name = "core", .module = core } } });
    if (metal) |module| root.addImport("metal", module);
    const exe = b.addExecutable(.{ .name = "bench-worker", .root_module = root });
    b.step("bench-worker", "Build the resident Engine Protocol v1 worker").dependOn(&b.addInstallArtifact(exe, .{}).step);
}

test {
    _ = dist_build;
}
