const std = @import("std");
const assert = std.debug.assert;
const expect = std.testing.expect;
const dimens = @import("dimens.zig");
const tensor = @import("tensor.zig");

pub fn Type(dims: []const usize) type {
    return [dimens._max(dims) + 1]usize;
}

pub fn initSize(comptime dims: []const usize) Type(dims) {
    return .{0} ** (dimens._max(dims) + 1);
}

pub fn collectSize(size: []usize, a: anytype) void {
    const A = @TypeOf(a);
    if (!tensor.is(A)) return;
    inline for (A.dims, 0..) |dim, i| {
        if (size[dim] == 0) {
            size[dim] = a.size.vec[i];
        } else {
            assert(size[dim] == a.size.vec[i]);
        }
    }
}

pub fn reset(running: []usize, comptime dims: []const usize) void {
    inline for (dims) |dim| {
        running[dim] = 0;
    }
}

pub fn next(running: []usize, size: []const usize, comptime dims: []const usize) bool {
    inline for (dims) |dim| {
        if (running[dim] == size[dim] - 1) {
            running[dim] = 0;
        } else {
            running[dim] += 1;
            return true;
        }
    }
    return false;
}

test "coord iteration" {
    const dims: []const usize = &.{ 0, 2 };
    const size: [3]usize = .{ 3, 4, 5 };
    var i: [3]usize = undefined;
    reset(&i, dims);
    try expect(i[0] == 0);
    try expect(i[2] == 0);
    for (0..4) |_| {
        try expect(next(&i, &size, dims));
    }
    try expect(i[0] == 1);
    try expect(i[2] == 1);
    for (0..10) |_| {
        try expect(next(&i, &size, dims));
    }
    try expect(!next(&i, &size, dims));
}
