const std = @import("std");
const expect = std.testing.expect;
const assert = std.debug.assert;
const dim = @import("dimens.zig");

/// n-dimensional position
pub fn Type(comptime dims: []const usize) type {
    return packed struct {
        pub const Position = @This();
        vec: @Vector(dims.len, usize),

        pub fn from(coord: []const usize) Position {
            assert(dims.len == 0 or coord.len > dim._max(dims)); //not enough coordinates provided
            var res: Position = undefined;
            inline for (dims, 0..) |d, i| {
                res.vec[i] = coord[d];
            }
            return res;
        }

        /// increment
        pub fn inc(size: Position) Position {
            var res: Position = undefined;
            if (dims.len == 0) return res;
            res.vec[0] = 1;
            inline for (0..dims.len - 1) |i| {
                res.vec[i + 1] = res.vec[i] * size.vec[i];
            }
            return res;
        }

        /// index from position and increment
        pub fn ind(increment: Position, pos: Position) usize {
            if (dims.len == 0) return 0;
            return @reduce(.Add, increment.vec * pos.vec);
        }

        /// all less than
        pub fn lt(pos: Position, size: Position) bool {
            if (dims.len == 0) return true;
            return @reduce(.And, pos.vec < size.vec);
        }

        /// number of entries
        pub fn cnt(size: Position) usize {
            return if (dims.len > 0) @reduce(.Mul, size.vec) else 1;
        }
    };
}
