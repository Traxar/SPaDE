const std = @import("std");
const expect = std.testing.expect;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const util = @import("util.zig");
const dim = @import("dimens.zig");
const position = @import("position.zig");
const coords = @import("coords.zig");
const args = @import("args.zig");
const simd = @import("simd.zig");

/// true if T is a Tensor
pub inline fn is(T: type) bool {
    comptime {
        if (@typeInfo(T) != .Struct) return false;
        if (!@hasDecl(T, "Element") or !@hasDecl(T, "dims")) return false;
        if (@TypeOf(T.dims) != []const usize) return false;
        return T == DenseType(T.Element, T.dims);
    }
}

pub fn Type(Element: type) type {
    return packed struct {
        pub fn Dense(comptime dimensions: []const usize) type {
            return DenseType(Element, dimensions);
        }

        //TODO: pub fn Sparse()

        pub fn f(comptime red: anytype, comptime ew: anytype, anyargs: anytype) args.Return(ew, @TypeOf(anyargs), Element) {
            const val = util.MultiSlice(Element)._init(1);
            const res = Dense(&.{}){
                .size = undefined,
                .incr = undefined,
                .offset = 0,
                .vals = val,
            };
            res.f(red, ew, anyargs);
            return res.at(&.{});
        }
    };
}

fn collectDims(Res: type, AnyArgs: type) []const usize {
    assert(is(util.Deref(Res)));
    return dim._union(util.Deref(Res).dims, args.collectDims(AnyArgs));
}

fn collectSize(res: anytype, anyargs: anytype) coords.Type(collectDims(@TypeOf(res), @TypeOf(anyargs))) {
    assert(is(util.Deref(@TypeOf(res))));
    var size = coords.initSize(collectDims(@TypeOf(res), @TypeOf(anyargs)));
    coords.collectSize(&size, res);
    args.collectSize(&size, anyargs);
    return size;
}

fn DenseType(Elem: type, comptime dimensions: []const usize) type {
    if (!dim._isSet(dimensions)) @compileError("dimensions must be unique");
    return packed struct {
        const Tensor = @This();
        pub const Element = Elem;
        pub const dims = dimensions;

        const Position = position.Type(dims);
        size: Position,
        incr: Position,
        offset: usize,
        vals: util.MultiSlice(Element),

        pub inline fn init(size: []const usize, allocator: Allocator) !Tensor {
            const sz = Position.from(size);
            assert(sz.cnt() > 0);
            return Tensor{
                .vals = try util.MultiSlice(Element).init(sz.cnt(), allocator),
                .incr = sz.inc(),
                .offset = 0,
                .size = sz,
            };
        }

        pub fn deinit(a: Tensor, allocator: Allocator) void {
            a.vals.deinit(allocator);
        }

        pub fn ensureCapacity(a: *Tensor, n: usize, allocator: Allocator) !void {
            try a.vals.ensureCapacity(n, allocator);
        }

        fn ind(a: Tensor, pos: Position) usize {
            return a.offset + a.incr.ind(pos);
        }

        pub fn at(a: Tensor, coord: []const usize) Element {
            const pos = Position.from(coord);
            assert(pos.lt(a.size)); //out of bounds
            return a.vals.at(a.ind(pos));
        }

        pub fn set(a: Tensor, coord: []const usize, element: Element) void {
            const pos = Position.from(coord);
            assert(pos.lt(a.size)); //out of bounds
            a.vals.set(a.ind(pos), element);
        }

        /// lazy swap of dimensions i and j
        pub fn t(a: Tensor, comptime i: usize, comptime j: usize) DenseType(Element, dim._swap(dims, i, j)) {
            return @bitCast(a);
        }

        /// lazily restrict the coordinate in dim d of a tensor a
        pub fn sub(a: Tensor, comptime d: usize, start: usize, size: usize) Tensor {
            const i = dim._index(dims, d).?; //d must be in dims
            assert(start + size <= a.size.vec[i]); //out of bounds
            var res = a;
            res.size.vec[i] = size;
            res.offset += start * a.incr.vec[i];
            return res;
        }

        /// lazily fix the coordinate in dim d of a tensor a
        /// this returns a tensor of lower dimension
        pub fn fix(a: Tensor, comptime d: usize, coord: usize) DenseType(Element, dim._sub(dims, &.{d})) {
            const i = dim._index(dims, d).?; //d must be in dims
            assert(coord <= a.size.vec[i]); //out of bounds
            var res: DenseType(Element, dim._sub(dims, &.{d})) = undefined;
            res.vals = a.vals;
            res.offset = a.offset + coord * a.incr.vec[i];
            inline for (0..dims.len - 1) |j| { //TODO use shuffle
                res.size.vec[j] = a.size.vec[if (j < i) j else j + 1];
                res.incr.vec[j] = a.incr.vec[if (j < i) j else j + 1];
            }
            return res;
        }

        /// res <- red(ew(anyargs))
        pub fn f(res: Tensor, comptime red: anytype, comptime ew: anytype, anyargs: anytype) args.Return(ew, @TypeOf(anyargs), void) {
            const AnyArgs = @TypeOf(anyargs);
            const dims_args = args.collectDims(AnyArgs);
            const dims_total = dim._union(dims, dims_args);
            return res.fInternal(red, ew, anyargs, dims_total);
        }

        /// TODO: incooperate simd.zig whereever possible
        fn fInternal(res: Tensor, comptime red: anytype, comptime ew: anytype, anyargs: anytype, comptime dims_order: []const usize) args.Return(ew, @TypeOf(anyargs), void) {
            const AnyArgs = @TypeOf(anyargs);
            const dims_args = args.collectDims(AnyArgs);
            const dims_total = dim._union(dims, dims_args);
            assert(dim._equal(dims_total, dims_order)); //dims_order must match dimensions of the tensor operation
            const dims_fill = dim._sub(dims_order, dims_args);
            const dims_red = dim._sub(dims_order, dims);
            const dims_calc = dim._sub(dim._sub(dims_order, dims_fill), dims_red);
            assert(args.validInplace(res, anyargs)); //inplace operation not possible
            const size = collectSize(res, anyargs);
            var a = args.init(anyargs);
            var i: coords.Type(dims_total) = undefined;
            coords.reset(&i, dims_calc);
            while (true) {
                coords.reset(&i, dims_red);
                args.set(anyargs, &a, &i);
                const res_red_err = @call(.auto, ew, a);
                var res_red = if (util.ErrorSet(ew, @TypeOf(a))) |_| try res_red_err else res_red_err;
                while (dims_red.len > 0 and coords.next(&i, &size, dims_red)) {
                    args.set(anyargs, &a, &i);
                    const res_ew_err = @call(.auto, ew, a);
                    const res_ew = if (util.ErrorSet(ew, @TypeOf(a))) |_| try res_ew_err else res_ew_err;
                    res_red = red(res_red, res_ew);
                }
                coords.reset(&i, dims_fill);
                while (true) {
                    res.set(&i, res_red);
                    if (!coords.next(&i, &size, dims_fill)) break;
                }
                if (!coords.next(&i, &size, dims_calc)) break;
            }
        }
    };
}

test "tensor type" {
    const ally = std.testing.allocator;
    const S = Type(f32).Dense(&.{ 1, 2 });
    const s = try S.init(&.{ 3, 4, 5 }, ally); //unused dimension 0 is ignored
    defer s.deinit(ally);
    try expect(s.size.vec[0] == 4);
    try expect(s.size.vec[1] == 5);
    try expect(s.vals.len == 20);
}

test "tensor type 0D" {
    const ally = std.testing.allocator;
    const S = Type(f32).Dense(&.{});
    const s = try S.init(&.{}, ally); //unused dimension 0 is ignored
    defer s.deinit(ally);
    s.set(&.{ 0, 1, 2 }, 3.5);
    try expect(s.vals.len == 1);
    try expect(s.at(&.{ 6, 7, 8 }) == 3.5);
}

test "tensor at/set" {
    const ally = std.testing.allocator;
    const S = Type(f32).Dense(&.{ 0, 1 });
    const s = try S.init(&.{ 2, 3 }, ally);
    defer s.deinit(ally);
    s.set(&.{ 0, 2, 4 }, 6);
    try expect(s.at(&.{ 0, 2 }) == 6);
}

test "tensor ew" {
    const op = @import("op.zig");
    const ally = std.testing.allocator;
    const T = Type(f32);
    const V = T.Dense(&.{0});
    const M = T.Dense(&.{ 0, 1 });
    const a = try V.init(&.{2}, ally);
    defer a.deinit(ally);
    a.set(&.{0}, 3);
    a.set(&.{1}, 4);
    const b = try V.init(&.{3}, ally);
    defer b.deinit(ally);
    b.set(&.{0}, 3);
    b.set(&.{1}, 2);
    b.set(&.{2}, 0);
    const c = try M.init(&.{ 2, 3 }, ally);
    defer c.deinit(ally);

    a.f(undefined, op.add, .{ a, 1 }); //inplace

    c.f(undefined, op.mul, .{ a, b.t(0, 1) });

    try expect(c.at(&.{ 0, 0 }) == 12);
    try expect(c.at(&.{ 0, 1 }) == 8);
    try expect(c.at(&.{ 0, 2 }) == 0);
    try expect(c.at(&.{ 1, 0 }) == 15);
    try expect(c.at(&.{ 1, 1 }) == 10);
    try expect(c.at(&.{ 1, 2 }) == 0);

    var d = T.f(op.add, op.id, .{c});
    try expect(d == 45);

    d = T.f(op.add, op.mul, .{ a, b.t(0, 1) });
    try expect(d == 45);
}

test "tensor sub/fix" {
    const ally = std.testing.allocator;
    const M = Type(f32).Dense(&.{ 0, 1 });
    const a = try M.init(&.{ 2, 3 }, ally);
    defer a.deinit(ally);
    a.set(&.{ 0, 0 }, 1);
    a.set(&.{ 0, 1 }, 2);
    a.set(&.{ 0, 2 }, 3);
    a.set(&.{ 1, 0 }, 4);
    a.set(&.{ 1, 1 }, 5);
    a.set(&.{ 1, 2 }, 6);
    const sub = a.sub(1, 1, 2);
    try expect(sub.vals.ptr == a.vals.ptr);
    try expect(sub.offset == 2);
    try expect(sub.at(&.{ 0, 0 }) == 2);
    try expect(sub.at(&.{ 0, 1 }) == 3);
    try expect(sub.at(&.{ 1, 0 }) == 5);
    try expect(sub.at(&.{ 1, 1 }) == 6);

    const fix = a.fix(0, 1);
    const V = @TypeOf(fix);
    try expect(V.dims.len == 1);
    try expect(V.dims[0] == 1);
    try expect(fix.vals.ptr == a.vals.ptr);
    try expect(fix.offset == 1);
    try expect(fix.size.vec[0] == 3);
    try expect(fix.at(&.{ 100, 0 }) == 4);
    try expect(fix.at(&.{ 100, 1 }) == 5);
    try expect(fix.at(&.{ 100, 2 }) == 6);
}
