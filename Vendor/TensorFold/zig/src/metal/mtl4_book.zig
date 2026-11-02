//! Pure bookkeeping of the Metal 4 command path (mtl4.zig), kept free of Objective-C so it tests on any host: the order
//! a command buffer's work and its event waits and signals reach the queue, the upload chunks its constants take, and
//! when a slot may be reused.
const std = @import("std");

/// One step of a command buffer as the queue receives it at commit, in the order the caller encoded it.
pub fn Item(comptime Handle: type) type {
    return union(enum) {
        /// An ended Metal 4 command buffer: committed on its own (`commit:count:` of one).
        cb: Handle,
        /// `waitForEvent:value:` on the queue before the next committed buffer.
        wait: struct { event: Handle, value: u64 },
        /// `signalEvent:value:` on the queue after the buffers committed before it.
        signal: struct { event: Handle, value: u64 },
    };
}

/// The steps of one wrapper command buffer: its encoded work split at every event wait or signal, so each reaches the
/// queue exactly where the caller placed it between encoders (Metal 4 waits and signals on the queue, not in a buffer).
pub fn Plan(comptime Handle: type) type {
    return struct {
        const Self = @This();
        items: std.ArrayListUnmanaged(Item(Handle)) = .empty,
        /// The open Metal 4 buffer, begun on the first encoder after a split (null: none open).
        open: ?Handle = null,
        /// Whether `open` holds any encoder: an empty one is ended and dropped, never committed.
        open_work: bool = false,

        /// The open buffer gets an encoder (`begin` made it when none was open).
        pub fn work(p: *Self, begin: anytype) !Handle {
            if (p.open == null) p.open = try begin.make();
            p.open_work = true;
            return p.open.?;
        }

        /// Ends the open buffer (`end.close(h)`), keeping it when it holds work; `end.drop(h)` for an empty one.
        pub fn close(p: *Self, a: std.mem.Allocator, end: anytype) !void {
            const h = p.open orelse return;
            end.close(h);
            if (p.open_work) try p.items.append(a, .{ .cb = h }) else end.drop(h);
            p.open = null;
            p.open_work = false;
        }

        pub fn wait(p: *Self, a: std.mem.Allocator, end: anytype, event: Handle, value: u64) !void {
            try p.close(a, end);
            try p.items.append(a, .{ .wait = .{ .event = event, .value = value } });
        }

        pub fn signal(p: *Self, a: std.mem.Allocator, end: anytype, event: Handle, value: u64) !void {
            try p.close(a, end);
            try p.items.append(a, .{ .signal = .{ .event = event, .value = value } });
        }

        pub fn reset(p: *Self) void {
            p.items.clearRetainingCapacity();
            p.open = null;
            p.open_work = false;
        }

        pub fn deinit(p: *Self, a: std.mem.Allocator) void {
            p.items.deinit(a);
        }
    };
}

/// Fixed-size upload chunks of one shared buffer, handed to command buffers whole and returned whole once the buffer
/// that took them completed. A command buffer's constants (setBytes) are bump-allocated inside its own chunks, so no
/// two buffers ever share a chunk and a chunk is reused only after the GPU finished reading it.
pub const Chunks = struct {
    pub const size: usize = 64 << 10;
    /// Constants are placed at this alignment (every Metal constant type, float4x4 included).
    pub const alignment: usize = 64;
    free: std.ArrayListUnmanaged(u32) = .empty,
    count: u32,

    pub fn init(a: std.mem.Allocator, count: u32) !Chunks {
        var c = Chunks{ .count = count };
        try c.free.ensureTotalCapacity(a, count);
        var i = count;
        while (i > 0) {
            i -= 1;
            c.free.appendAssumeCapacity(i);
        }
        return c;
    }

    pub fn take(c: *Chunks) ?u32 {
        return c.free.pop();
    }

    pub fn give(c: *Chunks, chunk: u32) void {
        std.debug.assert(chunk < c.count);
        c.free.appendAssumeCapacity(chunk);
    }

    pub fn deinit(c: *Chunks, a: std.mem.Allocator) void {
        c.free.deinit(a);
    }
};

/// A command buffer's place in its chunks: the chunk it fills and how far.
pub const Cursor = struct {
    chunk: ?u32 = null,
    used: usize = 0,

    /// The byte offset in the upload buffer for `len` bytes, or null when the current chunk has no room (the caller
    /// takes a new chunk and calls `start`). `len` above a chunk is refused by the caller (setBytes holds at most 4 KB).
    pub fn place(c: *Cursor, len: usize) ?usize {
        const chunk = c.chunk orelse return null;
        const at = std.mem.alignForward(usize, c.used, Chunks.alignment);
        if (at + len > Chunks.size) return null;
        c.used = at + len;
        return @as(usize, chunk) * Chunks.size + at;
    }

    pub fn start(c: *Cursor, chunk: u32) void {
        c.chunk = chunk;
        c.used = 0;
    }
};

/// A handle to a reusable slot: a stale handle (its slot reused since) names a buffer that completed, because a slot is
/// reused only after the buffer it held completed.
pub const Ticket = struct {
    generation: u32,

    pub fn current(t: Ticket, slot_generation: u32) bool {
        return t.generation == slot_generation;
    }
};

/// Whether timeline value `seen` shows epoch `epoch` complete (0: not committed yet).
pub fn done(epoch: u64, seen: u64) bool {
    return epoch != 0 and seen >= epoch;
}

const testing = std.testing;

const Fake = struct {
    next: u32 = 100,
    closed: std.ArrayListUnmanaged(u32) = .empty,
    dropped: std.ArrayListUnmanaged(u32) = .empty,
    fn make(f: *Fake) !u32 {
        f.next += 1;
        return f.next;
    }
    fn close(f: *Fake, h: u32) void {
        f.closed.append(testing.allocator, h) catch unreachable;
    }
    fn drop(f: *Fake, h: u32) void {
        f.dropped.append(testing.allocator, h) catch unreachable;
    }
};

test "waits and signals split the buffer where the caller placed them" {
    var f = Fake{};
    defer f.closed.deinit(testing.allocator);
    defer f.dropped.deinit(testing.allocator);
    var p = Plan(u32){};
    defer p.deinit(testing.allocator);
    // wait e1 1; encode; signal s1 5; wait e2 2; encode; encode; signal s2 6; commit
    try p.wait(testing.allocator, &f, 7, 1);
    _ = try p.work(&f);
    try p.signal(testing.allocator, &f, 8, 5);
    try p.wait(testing.allocator, &f, 9, 2);
    const b = try p.work(&f);
    try testing.expectEqual(b, try p.work(&f)); // a second encoder joins the open buffer
    try p.signal(testing.allocator, &f, 10, 6);
    try p.close(testing.allocator, &f);
    const I = Item(u32);
    const want = [_]I{ .{ .wait = .{ .event = 7, .value = 1 } }, .{ .cb = 101 }, .{ .signal = .{ .event = 8, .value = 5 } }, .{ .wait = .{ .event = 9, .value = 2 } }, .{ .cb = 102 }, .{ .signal = .{ .event = 10, .value = 6 } } };
    try testing.expectEqual(want.len, p.items.items.len);
    for (want, p.items.items) |w, got| try testing.expect(std.meta.eql(w, got));
    try testing.expectEqual(@as(usize, 0), f.dropped.items.len);
}

test "an empty open buffer is dropped, never committed" {
    var f = Fake{};
    defer f.closed.deinit(testing.allocator);
    defer f.dropped.deinit(testing.allocator);
    var p = Plan(u32){};
    defer p.deinit(testing.allocator);
    p.open = try f.make(); // begun but no encoder
    try p.signal(testing.allocator, &f, 3, 1);
    try p.close(testing.allocator, &f);
    try testing.expectEqual(@as(usize, 1), p.items.items.len);
    try testing.expect(p.items.items[0] == .signal);
    try testing.expectEqual(@as(usize, 1), f.dropped.items.len);
}

test "upload cursor aligns, stays inside its chunk and asks for a new one" {
    var c = Cursor{};
    try testing.expectEqual(@as(?usize, null), c.place(16));
    c.start(3);
    try testing.expectEqual(@as(?usize, 3 * Chunks.size), c.place(12));
    try testing.expectEqual(@as(?usize, 3 * Chunks.size + 64), c.place(4));
    try testing.expectEqual(@as(?usize, 3 * Chunks.size + 128), c.place(4096));
    // fill to the end: the last placement that fits ends exactly at the chunk's end
    var last: usize = 0;
    while (c.place(64)) |at| last = at;
    try testing.expectEqual(@as(usize, 3 * Chunks.size + Chunks.size - 64), last);
    try testing.expectEqual(@as(?usize, null), c.place(1));
    c.start(0);
    try testing.expectEqual(@as(?usize, 0), c.place(1));
}

test "chunks are handed out once and come back" {
    var c = try Chunks.init(testing.allocator, 4);
    defer c.deinit(testing.allocator);
    var seen: [4]bool = @splat(false);
    for (0..4) |_| {
        const k = c.take().?;
        try testing.expect(!seen[k]);
        seen[k] = true;
    }
    try testing.expectEqual(@as(?u32, null), c.take());
    c.give(2);
    try testing.expectEqual(@as(?u32, 2), c.take());
}

test "epochs complete only once committed and reached" {
    try testing.expect(!done(0, 99));
    try testing.expect(!done(5, 4));
    try testing.expect(done(5, 5));
    try testing.expect(done(5, 9));
    const t = Ticket{ .generation = 3 };
    try testing.expect(t.current(3));
    try testing.expect(!t.current(4));
}
