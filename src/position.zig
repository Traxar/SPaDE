const std = @import("std");
const expect = std.testing.expect;
const assert = std.debug.assert;
const Dims = @import("dims.zig").Type;

/// n-dimensional Index
pub fn Type(Index: type, dims: Dims) type {
    if (@typeInfo(Index).int.signedness != .unsigned) @compileError("Index must be an unsigned integer");
    return packed struct {
        pub const Position = @This();
        pub const zero = Position{ .vec = @splat(0) };
        vec: @Vector(dims.len, Index),

        pub fn from(coord: []const Index) Position {
            if (dims.len == 0) return undefined;
            assert(coord.len > dims.max()); //not enough coordinates provided
            const mask = comptime _: {
                var mask: @Vector(dims.len, i32) = undefined;
                for (dims.slice(), 0..) |d, i| {
                    mask[i] = d;
                }
                break :_ mask;
            };
            return .{ .vec = @shuffle(Index, coord[0 .. dims.max() + 1].*, undefined, mask) };
        }

        /// value at dimension `d`, `null` if `d` does not exist
        pub inline fn at(a: Position, comptime d: usize) ?Index {
            return if (dims.index(d)) |i| a.vec[i] else null;
        }

        /// set value at dimension
        pub fn set(a: *Position, comptime d: usize, new: Index) void {
            a.vec[dims.index(d).?] = new;
        }

        /// increment from size
        pub fn increment(size: Position) Position {
            assert(zero.lt(size));
            var res: Position = undefined;
            if (dims.len == 0) return res;
            res.vec[0] = 1;
            inline for (0..dims.len - 1) |i| {
                res.vec[i + 1] = res.vec[i] * size.vec[i];
            }
            return res;
        }

        /// index from increment and position
        pub fn index(incr: Position, pos: Position) Index {
            if (dims.len == 0) return 0;
            return @reduce(.Add, incr.vec * pos.vec);
        }

        fn increasing(incr: Position) bool {
            switch (dims.len) {
                0, 1 => return true,
                else => {
                    const lower = incr.cut(dims.ptr[dims.len - 1]).vec;
                    const upper = incr.cut(dims.ptr[0]).vec;
                    return @reduce(.And, lower <= upper);
                },
            }
        }

        /// position from increment and index
        pub fn position(incr: Position, ind: Index) ?Position {
            assert(incr.increasing());
            var pos = Position{ .vec = undefined };
            if (dims.len == 0) return pos;
            var i = ind;
            var j: usize = dims.len;
            while (j > 0) {
                j -= 1;
                pos.vec[j] = @divFloor(i, incr.vec[j]);
                i -= pos.vec[j] * incr.vec[j];
            }
            if (i != 0) return null;
            assert(incr.index(pos) == ind);
            return pos;
        }

        /// all less than
        pub fn lt(a: Position, b: Position) bool {
            if (dims.len == 0) return true;
            return @reduce(.And, a.vec < b.vec);
        }

        /// all equal
        fn eq(a: Position, b: Position) bool {
            if (dims.len == 0) return true;
            return @reduce(.And, a.vec == b.vec);
        }

        /// multiply entries
        pub fn mul(size: Position) Index {
            return if (dims.len > 0) @reduce(.Mul, size.vec) else 1;
        }

        /// remove dimension d
        pub fn cut(a: Position, comptime d: usize) Type(Index, dims.sub(Dims.from(&.{d}))) {
            const i = dims.index(d);
            assert(i != null); //d must be in dims
            const mask = comptime _: {
                var m: @Vector(dims.len - 1, i32) = undefined;
                for (0..dims.len - 1) |j| {
                    m[j] = if (j < i.?) j else j + 1;
                }
                break :_ m;
            };
            return .{ .vec = @shuffle(Index, a.vec, undefined, mask) };
        }

        pub fn next(iter: *Position, size: Position) bool {
            inline for (dims.slice()) |dim| {
                if (iter.at(dim) orelse 0 == size.at(dim).? - 1) {
                    iter.set(dim, 0);
                } else {
                    iter.set(dim, (iter.at(dim) orelse 0) + 1);
                    return true;
                }
            }
            return false;
        }
    };
}

test "position from index" {
    const Pos = Type(u16, Dims.from(&.{ 0, 1, 2 }));
    const size = Pos.from(&.{ 2, 3, 4 });
    const incr = size.increment();
    try expect(incr.eq(Pos.from(&.{ 1, 2, 6 })));
    const pos = Pos.from(&.{ 1, 1, 2 });
    const ind = incr.index(pos);
    try expect(ind == 15);
    const pos_ = incr.position(ind).?;
    try expect(pos.eq(pos_));
}
