const std = @import("std");
const expect = std.testing.expect;
const assert = std.debug.assert;

pub const Type = struct {
    const Dims = @This();
    ptr: [*]const usize,
    len: comptime_int,

    pub inline fn from(comptime dims: []const usize) Dims {
        comptime {
            const res = Dims{ .ptr = dims.ptr, .len = dims.len };
            if (!res.isSet()) @compileError("dimensions must be unique");
            return res;
        }
    }

    /// comptime-only check if each element is unique
    inline fn isSet(dims: Dims) bool {
        comptime {
            for (dims.slice(), 0..) |d, j| {
                for (0..j) |i| {
                    if (dims.ptr[i] == d) return false;
                }
            }
            return true;
        }
    }

    test "isSet" {
        const a = from(&.{ 0, 1 });
        //const b = from(&.{ 0, 1, 0 });
        try expect(a.isSet());
        //try expect(!b.isSet());
    }

    pub inline fn slice(dims: Dims) []const usize {
        comptime {
            return dims.ptr[0..dims.len];
        }
    }

    /// comptime-only difference of 2 sets
    /// <- a - b
    /// order of a is prefered
    pub inline fn sub(a: Dims, b: Dims) Dims {
        comptime {
            assert(a.isSet());
            assert(b.isSet());
            var res: []const usize = &.{};
            outer: for (a.slice()) |a_i| {
                for (b.slice()) |b_j| {
                    if (a_i == b_j) continue :outer;
                }
                res = res ++ .{a_i};
            }
            return from(res);
        }
    }

    test "sub" {
        const a = from(&.{ 0, 1 });
        const b = from(&.{ 1, 2, 3 });
        const c = b.sub(a);
        try expect(c.len == 2);
        try expect(c.ptr[0] == 2);
        try expect(c.ptr[1] == 3);
    }

    /// comptime-only union of 2 sets
    /// - order of `a` is prefered
    pub inline fn unite(a: Dims, b: Dims) Dims {
        comptime {
            return from(a.slice() ++ b.sub(a).slice());
        }
    }

    test "unite" {
        const a = from(&.{ 0, 1 });
        const b = from(&.{ 1, 2, 3 });
        const c = a.unite(b);
        try expect(c.len == 4);
        try expect(c.ptr[0] == 0);
        try expect(c.ptr[1] == 1);
        try expect(c.ptr[2] == 2);
        try expect(c.ptr[3] == 3);
    }

    /// comptime-only intersection of 2 sets
    /// order of a is prefered
    pub inline fn intersect(a: Dims, b: Dims) Dims {
        comptime {
            return a.sub(a.sub(b));
        }
    }

    test "intersect" {
        const a = from(&.{ 0, 1 });
        const b = from(&.{ 1, 2, 3 });
        const c = a.intersect(b);
        try expect(c.len == 1);
        try expect(c.ptr[0] == 1);
    }

    ///comptime-only maximum element of a set
    pub inline fn max(a: Dims) usize {
        comptime {
            var res: usize = 0;
            for (a.slice()) |dim| {
                if (dim > res) res = dim;
            }
            return res;
        }
    }

    test "max" {
        const a = from(&.{ 1, 2, 3 });
        try expect(a.max() == 3);
    }

    /// comptime-only
    /// true if a contains dim b
    pub inline fn has(a: Dims, b: usize) bool {
        comptime {
            return a.index(b) != null;
        }
    }

    test "has" {
        const a = from(&.{ 0, 1 });
        try expect(a.has(0));
        try expect(a.has(1));
        try expect(!a.has(2));
        try expect(!a.has(3));
    }

    /// comptime-only
    /// true if a > b
    pub inline fn contains(a: Dims, b: Dims) bool {
        comptime {
            return b.sub(a).len == 0;
        }
    }

    test "contains" {
        const a = from(&.{ 0, 1 });
        const b = from(&.{0});
        try expect(a.contains(b));
        try expect(!b.contains(a));
    }

    /// comptime-only
    /// true if a == b
    pub inline fn equal(a: Dims, b: Dims) bool {
        comptime {
            return a.contains(b) and b.contains(a);
        }
    }

    test "equal" {
        const a = from(&.{ 0, 1 });
        const b = from(&.{0});
        try expect(a.equal(a));
        try expect(!b.equal(a));
    }

    /// comptime-only
    /// replace i by j and j by i
    pub inline fn swap(dims: Dims, i: usize, j: usize) Dims {
        comptime {
            var res: []const usize = &.{};
            for (dims.slice()) |d| {
                res = res ++ .{switch (d) {
                    i => j,
                    j => i,
                    else => d,
                }};
            }
            return from(res);
        }
    }

    test "swap" {
        const a = from(&.{ 0, 1 }).swap(0, 1);
        try expect(a.ptr[0] == 1);
        try expect(a.ptr[1] == 0);
        const b = from(&.{ 0, 1 }).swap(0, 2);
        try expect(b.ptr[0] == 2);
        try expect(b.ptr[1] == 1);
    }

    /// comptime-only
    /// index of i in a
    pub inline fn index(a: Dims, i: usize) ?usize {
        comptime {
            for (a.slice(), 0..) |a_j, j| {
                if (a_j == i) return j;
            }
            return null;
        }
    }

    test "index" {
        const a = from(&.{ 0, 2, 3 });
        try expect(a.index(0) == 0);
        try expect(a.index(1) == null);
        try expect(a.index(2) == 1);
        try expect(a.index(3) == 2);
    }
};

test {
    _ = Type;
}
