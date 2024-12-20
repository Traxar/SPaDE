const std = @import("std");
const expect = std.testing.expect;
const assert = std.debug.assert;

const Dims = []const usize;

pub fn _isSet(dims: Dims) bool {
    for (dims, 0..) |d, j| {
        for (0..j) |i| {
            if (dims[i] == d) return false;
        }
    }
    return true;
}

/// comptime-only difference of 2 sets
/// <- a - b
/// order of a is prefered
pub inline fn _difference(a: Dims, b: Dims) Dims {
    comptime {
        assert(_isSet(a));
        assert(_isSet(b));
        var res: Dims = &.{};
        outer: for (a) |a_i| {
            for (b) |b_j| {
                if (a_i == b_j) continue :outer;
            }
            res = res ++ .{a_i};
        }
        return res;
    }
}

test "diff" {
    const a: Dims = &.{ 0, 1 };
    const b: Dims = &.{ 1, 2, 3 };
    const c = _difference(b, a);
    try expect(c.len == 2);
    try expect(c[0] == 2);
    try expect(c[1] == 3);
}

/// comptime-only union of 2 sets
/// order of a is prefered
pub inline fn _union(a: Dims, b: Dims) Dims {
    return a ++ _difference(b, a);
}

test "combine" {
    const a: Dims = &.{ 0, 1 };
    const b: Dims = &.{ 1, 2, 3 };
    const c = _union(a, b);
    try expect(c.len == 4);
    try expect(c[0] == 0);
    try expect(c[1] == 1);
    try expect(c[2] == 2);
    try expect(c[3] == 3);
}

///comptime-only intersection of 2 sets
pub inline fn _intersect(dims: Dims, other: Dims) Dims {
    return _difference(dims, _difference(dims, other));
}

test "intersect" {
    const a: Dims = &.{ 0, 1 };
    const b: Dims = &.{ 1, 2, 3 };
    const c = _intersect(a, b);
    try expect(c.len == 1);
    try expect(c[0] == 1);
}

///comptime-only maximum element of a set
pub inline fn _max(dims: Dims) usize {
    comptime {
        var res: usize = 0;
        for (dims) |dim| {
            if (dim > res) res = dim;
        }
        return res;
    }
}

test "max" {
    try expect(_max(&.{ 1, 2, 3 }) == 3);
}

/// comptime-only
/// true if dims > other
pub inline fn _contains(dims: Dims, other: Dims) bool {
    comptime {
        return _difference(other, dims).len == 0;
    }
}

test "contains" {
    try expect(_contains(&.{ 0, 1 }, &.{0}));
    try expect(!_contains(&.{0}, &.{ 0, 1 }));
}

/// comptime-only
/// true if dims == other
pub inline fn _equal(dims: Dims, other: Dims) bool {
    comptime {
        return _contains(dims, other) and _contains(other, dims);
    }
}

test "equal" {
    try expect(_equal(&.{ 0, 1 }, &.{ 0, 1 }));
    try expect(!_equal(&.{0}, &.{ 0, 1 }));
}
