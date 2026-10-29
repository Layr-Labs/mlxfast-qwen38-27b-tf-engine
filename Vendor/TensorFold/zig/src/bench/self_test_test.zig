const std = @import("std");

const ShortRoot = struct {
    buffer: [64]u8 = undefined,
    len: usize = 0,

    fn init() !ShortRoot {
        var result: ShortRoot = .{};
        for (0..8) |_| {
            var random: [6]u8 = undefined;
            std.testing.io.random(&random);
            var encoded: [std.base64.url_safe.Encoder.calcSize(random.len)]u8 = undefined;
            _ = std.base64.url_safe.Encoder.encode(&encoded, &random);
            const root = try std.fmt.bufPrint(&result.buffer, "/tmp/tfb-{s}", .{encoded});
            std.Io.Dir.createDirAbsolute(std.testing.io, root, .fromMode(0o700)) catch |err| {
                if (err == error.PathAlreadyExists) continue;
                return err;
            };
            result.len = root.len;
            return result;
        }
        return error.PathAlreadyExists;
    }

    fn path(root: *const ShortRoot, name: []const u8) ![]u8 {
        return std.fs.path.join(std.testing.allocator, &.{ root.buffer[0..root.len], name });
    }

    fn cleanup(root: *ShortRoot) void {
        std.Io.Dir.cwd().deleteTree(std.testing.io, root.buffer[0..root.len]) catch {};
    }
};

pub fn Cases(comptime Paths: type, comptime prepare: anytype) type {
    return struct {
        fn cleanup(paths: *Paths) void {
            if (paths.dir_created) std.Io.Dir.cwd().deleteDir(std.testing.io, paths.dir) catch {};
            paths.deinit();
        }

        fn expectPrivateDirectory(path: []const u8) !void {
            const stat = try std.Io.Dir.cwd().statFile(std.testing.io, path, .{});
            try std.testing.expectEqual(@as(std.posix.mode_t, 0o700), stat.permissions.toMode() & 0o777);
        }

        test "self-test uses the requested directory" {
            var root = try ShortRoot.init();
            defer root.cleanup();
            const requested = try root.path("requested");
            defer std.testing.allocator.free(requested);
            var paths = try prepare(std.testing.allocator, std.testing.io, requested, "/ignored", 1, 2);
            defer cleanup(&paths);
            try std.testing.expectEqualStrings(requested, paths.dir);
            const expected_socket = try std.fs.path.join(std.testing.allocator, &.{ requested, "resident.sock" });
            defer std.testing.allocator.free(expected_socket);
            try std.testing.expectEqualStrings(expected_socket, paths.socket);
            try expectPrivateDirectory(paths.dir);
        }

        test "self-test creates its unique directory under TMPDIR" {
            var root = try ShortRoot.init();
            defer root.cleanup();
            const root_path = root.buffer[0..root.len];
            const with_slash = try std.fmt.allocPrint(std.testing.allocator, "{s}/", .{root_path});
            defer std.testing.allocator.free(with_slash);
            var paths = try prepare(std.testing.allocator, std.testing.io, null, with_slash, 12, 34);
            defer cleanup(&paths);
            try std.testing.expectEqualStrings(root_path, std.fs.path.dirname(paths.dir).?);
            try std.testing.expectEqualStrings("tfbw-12-34", std.fs.path.basename(paths.dir));
            try expectPrivateDirectory(paths.dir);
        }

        test "self-test refuses a symlink directory" {
            var root = try ShortRoot.init();
            defer root.cleanup();
            const target = try root.path("target");
            defer std.testing.allocator.free(target);
            const link = try root.path("link");
            defer std.testing.allocator.free(link);
            try std.Io.Dir.cwd().createDir(std.testing.io, target, .default_dir);
            try std.Io.Dir.cwd().symLink(std.testing.io, target, link, .{ .is_directory = true });
            try std.testing.expectError(error.SelfTestDirSymlink, prepare(std.testing.allocator, std.testing.io, link, null, 1, 2));
        }

        test "self-test refuses a socket path at the sun_path limit" {
            var long: [104 - "/resident.sock".len]u8 = @splat('x');
            long[0] = '/';
            try std.testing.expectError(error.SocketPathTooLong, prepare(std.testing.allocator, std.testing.io, &long, null, 1, 2));
        }
    };
}
