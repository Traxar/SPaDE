const std = @import("std");
const expect = std.testing.expect;
const assert = std.debug.assert;
const Dims = @import("dims.zig").Type;

/// n-dimensional Index
pub fn Type(comptime dims: Dims) type {
    return packed struct {
        pub const Position = @This();
        vec: @Vector(dims.len, usize),

        pub fn from(coord: []const usize) Position {
            if (dims.len == 0) return undefined;
            assert(coord.len > dims.max()); //not enough coordinates provided
            const mask = comptime _: {
                var m: @Vector(dims.len, i32) = undefined;
                for (dims.slice(), 0..) |d, i| {
                    m[i] = d;
                }
                break :_ m;
            };
            return .{ .vec = @shuffle(usize, coord[0 .. dims.max() + 1].*, undefined, mask) };
        }

        /// value at dimension
        pub fn at(pos: Position, comptime d: usize) usize {
            return pos.vec[dims.index(d).?];
        }

        /// set value at dimension
        pub fn set(pos: *Position, comptime d: usize, new: usize) void {
            pos.vec[dims.index(d).?] = new;
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
        pub fn mul(size: Position) usize {
            return if (dims.len > 0) @reduce(.Mul, size.vec) else 1;
        }

        /// remove dimension d
        pub fn cut(a: Position, comptime d: usize) Type(dims.sub(Dims.from(&.{d}))) {
            const i = dims.index(d);
            assert(i != null); //d must be in dims
            const mask = comptime _: {
                var m: @Vector(dims.len - 1, i32) = undefined;
                for (0..dims.len - 1) |j| {
                    m[j] = if (j < i.?) j else j + 1;
                }
                break :_ m;
            };
            return .{ .vec = @shuffle(usize, a.vec, undefined, mask) };
        }
    };
}
