const std = @import("std");
const expect = std.testing.expect;
const assert = std.debug.assert;
const Dims = @import("dims.zig").Type;
const PositionFrom = @import("position.zig").Type;

/// Returns `true` if `L` is a layout.
pub inline fn is(L: type) bool {
    comptime {
        if (@typeInfo(L) != .@"struct") return false;
        if (!@hasDecl(L, "dims")) return false;
        if (@TypeOf(L.dims) != Dims) return false;
        return L == Type(L.dims);
    }
}

pub fn Type(comptime _dims: Dims) type {
    return packed struct {
        pub const dims = _dims;
        const Layout = @This();
        const Position = PositionFrom(dims);
        size: Position,
        incr: Position,
        offset: usize,

        pub fn from(size: []const usize) Layout {
            const sz = Position.from(size);
            return .{
                .size = sz,
                .incr = sz.increment(),
                .offset = 0,
            };
        }

        /// get index of position `coords`
        pub fn index(layout: Layout, coords: []const usize) usize {
            return layout.indexFromPosition(Position.from(coords));
        }

        /// get index of position `pos`
        pub fn indexFromPosition(layout: Layout, pos: Position) usize {
            assert(pos.lt(layout.size)); //out of bounds
            return layout.offset + layout.incr.index(pos);
        }

        /// get position of index `ind` if existent else null
        fn position(layout: Layout, ind: usize) ?Position {
            if (layout.offset > ind) return null;
            const pos_ = layout.incr.position(ind - layout.offset) orelse return null;
            if (!pos_.lt(layout.size)) return null;
            return pos_;
        }

        /// number of entries
        pub fn n(layout: Layout) usize {
            return layout.size.mul();
        }

        /// Lazily swap dimensions `i` and `j`.
        /// - this has no cost
        pub fn t(layout: Layout, comptime i: usize, comptime j: usize) Type(dims.swap(i, j)) {
            return @bitCast(layout);
        }

        /// Restrict the coordinate of dimension `d` to size `size` starting at `start`.
        pub fn clamp(layout: Layout, comptime d: usize, start: usize, size: usize) Layout {
            assert(size > 0);
            assert(start + size <= layout.size.at(d)); //out of bounds
            var res = layout;
            res.size.set(d, size);
            res.offset += start * layout.incr.at(d);
            return res;
        }

        /// Fix the coordinate in dimension `d` to `coord`.
        /// - This returns a tensor of lower order.
        pub fn sub(layout: Layout, comptime d: usize, coord: usize) Type(dims.sub(Dims.from(&.{d}))) {
            assert(coord <= layout.size.at(d)); //out of bounds
            return .{
                .offset = layout.offset + coord * layout.incr.at(d),
                .size = layout.size.cut(d),
                .incr = layout.incr.cut(d),
            };
        }

        /// Take the diagonal of dimensions `i` and `j`.
        /// - The result is a tensor without dimension `j`.
        pub fn diag(layout: Layout, comptime i: usize, comptime j: usize) Type(dims.sub(Dims.from(&.{j}))) {
            assert(i != j);
            assert(dims.contains(Dims.from(&.{ i, j })));
            if (dims.index(i).? < dims.index(j).?) return layout.diag(j, i).t(i, j);
            var size = layout.size.cut(j);
            size.set(i, @min(layout.size.at(i), layout.size.at(j)));
            var incr = layout.incr.cut(j);
            incr.set(i, layout.incr.at(i) + layout.incr.at(j));
            return .{
                .offset = layout.offset,
                .size = size,
                .incr = incr,
            };
        }

        pub fn validInplace(layout: Layout, other: anytype, comptime dims_calc: Dims) bool {
            assert(dims_calc.len != 0);
            const L = @TypeOf(other);
            if (!is(L)) @compileError("`other` must be a Layout");
            const heuristic_layout = layout.n() * L.dims.len;
            const heuristic_other = other.n() * dims.len;
            return if (heuristic_layout <= heuristic_other)
                layout.checkValidInplace(other, dims_calc)
            else
                other.checkValidInplace(layout, dims_calc);
        }

        fn checkValidInplace(iter: Layout, check: anytype, comptime dims_calc: Dims) bool {
            assert(dims_calc.len != 0);
            const L = @TypeOf(check);
            if (!is(L)) @compileError("`other` must be a Layout");
            var pos_iter = Position.zero;
            while (true) {
                const ind = iter.indexFromPosition(pos_iter);
                if (check.position(ind)) |pos_check| {
                    inline for (dims_calc.slice()) |dim| {
                        if (pos_check.at(dim) != pos_iter.at(dim)) return false;
                    }
                }
                if (!pos_iter.next(iter.size)) return true;
            }
        }
    };
}

test "layout type" {
    const S = Type(Dims.from(&.{ 1, 2 }));
    const s = S.from(&.{ 3, 4, 5 }); //unused dimension 0 is ignored
    try expect(s.size.vec[0] == 4);
    try expect(s.size.vec[1] == 5);
}

test "layout 0D" {
    const S = Type(Dims.from(&.{}));
    const s = S.from(&.{});
    try expect(@TypeOf(s.size.vec) == @Vector(0, usize));
}

test "layout inplace" {
    const V = Type(Dims.from(&.{0}));
    const M = Type(Dims.from(&.{ 0, 1 }));
    const a = M.from(&.{ 4, 4 });

    try expect(a.validInplace(a, M.dims));
    try expect(!a.validInplace(a.t(0, 1), M.dims));

    const v = a.sub(0, 1).t(0, 1);
    try expect(v.validInplace(v, V.dims));
    try expect(v.validInplace(a.sub(1, 1), V.dims));
    try expect(!v.validInplace(a.sub(1, 2), V.dims));
    try expect(v.validInplace(a.sub(0, 2).t(0, 1), V.dims));

    try expect(v.clamp(0, 0, 3).validInplace(v.clamp(0, 0, 3), V.dims));
    try expect(!v.clamp(0, 1, 3).validInplace(v.clamp(0, 0, 3), V.dims));
    try expect(!v.clamp(0, 0, 3).validInplace(v.clamp(0, 1, 3), V.dims));
    try expect(v.clamp(0, 1, 3).validInplace(v.clamp(0, 1, 3), V.dims));

    const b = a.clamp(0, 1, 2).clamp(1, 0, 2);
    const c = a.clamp(0, 0, 2).clamp(1, 1, 2);
    try expect(!b.validInplace(c, M.dims));
    try expect(b.t(0, 1).validInplace(c, M.dims));

    const d = a.diag(0, 1);
    try expect(!a.validInplace(d, M.dims));
    try expect(a.validInplace(d, V.dims));
}

test "layout sub/fix" {
    const M = Type(Dims.from(&.{ 0, 1 }));
    const a = M.from(&.{ 2, 3 });

    const b = a.clamp(1, 1, 2);
    try expect(b.offset == 2);
    try expect(b.size.vec[0] == 2);
    try expect(b.size.vec[1] == 2);

    const c = a.sub(0, 1);
    const V = @TypeOf(c);
    try expect(V.dims.len == 1);
    try expect(V.dims.ptr[0] == 1);
    try expect(c.offset == 1);
    try expect(c.size.vec[0] == 3);
}
