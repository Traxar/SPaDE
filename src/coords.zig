const std = @import("std");
const assert = std.debug.assert;
const expect = std.testing.expect;
const Dims = @import("dims.zig").Type;
const tensor = @import("tensor.zig");

pub fn Type(dims: Dims) type {
    return struct {
        const Coords = @This();
        arr: [dims.max() + 1]usize,

        pub const zero = Coords{ .arr = .{0} ** (dims.max() + 1) };

        /// if `anyarg` is a tensor add/check its bounds.
        pub fn collectBounds(bounds: *Coords, anyarg: anytype) void {
            const AnyArg = @TypeOf(anyarg);
            if (!tensor.is(AnyArg)) return;
            inline for (AnyArg.dims.slice()) |dim| {
                if (bounds.arr[dim] == 0) {
                    bounds.arr[dim] = anyarg.size(dim);
                } else {
                    assert(bounds.arr[dim] == anyarg.size(dim)); //bounds must match
                }
            }
        }

        pub fn collectBoundsMany(bounds: *Coords, anyargs: anytype) void {
            const AnyArgs = @TypeOf(anyargs);
            inline for (@typeInfo(AnyArgs).@"struct".fields) |field_anyargs| {
                bounds.collectBounds(@field(anyargs, field_anyargs.name));
            }
        }

        pub fn reset(iter: *Coords, comptime dims_iter: Dims) void {
            inline for (dims_iter.slice()) |dim| {
                iter.arr[dim] = 0;
            }
        }

        pub fn next(iter: *Coords, size: Coords, comptime dims_iter: Dims) bool {
            inline for (dims_iter.slice()) |dim| {
                if (iter.arr[dim] == size.arr[dim] - 1) {
                    iter.arr[dim] = 0;
                } else {
                    iter.arr[dim] += 1;
                    return true;
                }
            }
            return false;
        }
    };
}

test "coord reset/next" {
    const dims = Dims.from(&.{ 0, 2 });
    const Coords = Type(dims);
    const size = Coords{
        .arr = .{ 3, 4, 5 },
    };
    var i = Coords.zero;

    for (0..4) |_| {
        try expect(i.next(size, dims));
    }
    try expect(i.arr[0] == 1);
    try expect(i.arr[1] == 0);
    try expect(i.arr[2] == 1);
    for (0..10) |_| {
        try expect(i.next(size, dims));
    }
    try expect(!i.next(size, dims));

    i.reset(dims);
    try expect(i.arr[0] == 0);
    try expect(i.arr[1] == 0);
    try expect(i.arr[2] == 0);
}
