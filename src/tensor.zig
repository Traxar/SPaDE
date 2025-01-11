const std = @import("std");
const expect = std.testing.expect;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const util = @import("util.zig");
const Dims = @import("dims.zig").Type;
const position = @import("position.zig");
const coords = @import("coords.zig");
const args = @import("args.zig");
const simd = @import("simd.zig");

/// true if T is a Tensor
pub inline fn is(T: type) bool {
    comptime {
        if (@typeInfo(T) != .Struct) return false;
        if (!@hasDecl(T, "Element") or !@hasDecl(T, "dims")) return false;
        if (@TypeOf(T.dims) != Dims) return false;
        return T == DenseType(T.Element, T.dims);
    }
}

pub fn Type(Element: type) type {
    return packed struct {
        /// dense tensor with dimensions `dims`
        pub fn Dense(comptime dims: []const usize) type {
            return DenseType(Element, Dims.from(dims));
        }

        //TODO: pub fn Sparse()

        /// <- red(ew(anyargs))
        /// used for operations that result in a scalar
        pub fn f(comptime red: anytype, comptime ew: anytype, anyargs: anytype) args.Type(@TypeOf(anyargs)).ErrorWrap(ew, Element) {
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

fn DenseType(_Element: type, comptime _dims: Dims) type {
    return packed struct {
        const Tensor = @This();
        pub const Element = _Element;
        pub const dims = _dims;

        const Position = position.Type(dims);
        size: Position,
        incr: Position,
        offset: usize,
        vals: util.MultiSlice(Element),

        pub fn init(size: []const usize, allocator: Allocator) !Tensor {
            const sz = Position.from(size);
            assert(sz.mul() > 0); // must alloc atleast 1 element
            return Tensor{
                .vals = try util.MultiSlice(Element).init(sz.mul(), allocator),
                .incr = sz.inc(),
                .offset = 0,
                .size = sz,
            };
        }

        /// invalidates data
        pub fn reinit(a: *Tensor, size: []const usize, allocator: Allocator) !void {
            const sz = Position.from(size);
            assert(sz.mul() > 0); // must alloc atleast 1 element
            try a.vals.ensureCapacity(sz.mul(), allocator);
            a.incr = sz.inc();
            a.offset = 0;
            a.size = sz;
        }

        pub fn deinit(a: Tensor, allocator: Allocator) void {
            a.vals.deinit(allocator);
        }

        /// position `pos` to index into `a.vals`
        fn ind(a: Tensor, pos: Position) usize {
            return a.offset + a.incr.ind(pos);
        }

        /// value of tensor `a` at coordinates `coord`
        pub fn at(a: Tensor, coord: []const usize) Element {
            const pos = Position.from(coord);
            assert(pos.lt(a.size)); //out of bounds
            return a.vals.at(a.ind(pos));
        }

        /// set value of tensor `a` at coordinates `coord`
        pub fn set(a: Tensor, coord: []const usize, element: Element) void {
            const pos = Position.from(coord);
            assert(pos.lt(a.size)); //out of bounds
            a.vals.set(a.ind(pos), element);
        }

        /// lazily swap dimensions `i` and `j`
        pub fn t(a: Tensor, comptime i: usize, comptime j: usize) DenseType(Element, dims.swap(i, j)) {
            return @bitCast(a);
        }

        /// lazily restrict the coordinate in dimension `d` of tensor `a`
        pub fn clamp(a: Tensor, comptime d: usize, start: usize, size: usize) Tensor {
            const i = dims.index(d).?; //d must be in dims
            assert(start + size <= a.size.vec[i]); //out of bounds
            var res = a;
            res.size.vec[i] = size;
            res.offset += start * a.incr.vec[i];
            return res;
        }

        /// lazily fix the coordinate in diminsion `d` of tensor `a`
        /// this returns a tensor of lower dimension
        pub fn sub(a: Tensor, comptime d: usize, coord: usize) DenseType(Element, dims.sub(Dims.from(&.{d}))) {
            const i = dims.index(d).?; //d must be in dims
            assert(coord <= a.size.vec[i]); //out of bounds
            return .{
                .vals = a.vals,
                .offset = a.offset + coord * a.incr.vec[i],
                .size = a.size.cut(d),
                .incr = a.incr.cut(d),
            };
        }

        pub fn diag(a: Tensor, comptime i: usize, comptime j: usize) DenseType(Element, dims.sub(Dims.from(&.{j}))) {
            var size = a.size.cut(j);
            size.set(i, @min(a.size.at(i), a.size.at(j)));
            var incr = a.incr.cut(j);
            incr.set(i, a.incr.at(i) + a.incr.at(j));
            return .{
                .vals = a.vals,
                .offset = a.offset,
                .size = size,
                .incr = incr,
            };
        }

        /// res <- red(ew(anyargs))
        pub fn f(res: Tensor, comptime red: anytype, comptime ew: anytype, anyargs: anytype) args.Type(@TypeOf(anyargs)).ErrorWrap(ew, void) {
            const Args = args.Type(@TypeOf(anyargs));
            return res.fInternal(red, ew, anyargs, dims.unite(Args.dims));
        }

        /// res <- red(ew(anyargs))
        /// with iteration order given by `dims_order`
        fn fInternal(res: Tensor, comptime red: anytype, comptime ew: anytype, anyargs: anytype, comptime dims_order: Dims) args.Type(@TypeOf(anyargs)).ErrorWrap(ew, void) {
            const Args = args.Type(@TypeOf(anyargs));
            const dims_total = dims.unite(Args.dims);
            assert(dims_total.equal(dims_order)); //dims_order must match dimensions of the tensor operation
            const dims_fill = dims_order.sub(Args.dims);
            const dims_red = dims_order.sub(dims);
            const dims_calc = dims_order.sub(dims_fill).sub(dims_red);
            assert(validInplace(res, anyargs)); //inplace operation not valid
            const size = res.collectSize(anyargs);
            var a = Args.init(anyargs);
            var i: coords.Type(dims_total) = undefined;
            i.reset(dims_calc);
            while (true) {
                i.reset(dims_red);
                a.set(anyargs, &i.arr);
                const res_red_err = a.call(ew);
                var res_red: Element = if (util.ErrorSet(@TypeOf(res_red_err))) |_| try res_red_err else res_red_err;
                while (dims_red.len > 0 and i.next(size, dims_red)) {
                    a.set(anyargs, &i.arr);
                    const res_ew_err = a.call(ew);
                    const res_ew: Element = if (util.ErrorSet(@TypeOf(res_ew_err))) |_| try res_ew_err else res_ew_err;
                    res_red = red(res_red, res_ew);
                }
                i.reset(dims_fill);
                while (true) {
                    res.set(&i.arr, res_red);
                    if (!i.next(size, dims_fill)) break;
                }
                if (!i.next(size, dims_calc)) break;
            }
        }

        /// union of all dimensions of `@This()` and `AnyArgs`
        inline fn collectDims(AnyArgs: type) Dims {
            comptime {
                return dims.unite(args.Type(AnyArgs).dims);
            }
        }

        /// total size of tensor `res` and arguments `anyargs`
        /// asserts matching sizes
        fn collectSize(res: Tensor, anyargs: anytype) coords.Type(collectDims(@TypeOf(anyargs))) {
            var size = coords.Type(collectDims(@TypeOf(anyargs))).zero;
            size.collectSize(res);
            size.collectSizeMany(anyargs);
            return size;
        }

        /// TODO: improve
        /// as of now it is to conservative when using sub tensors
        /// and incorrect when manipulating the underlying MultiSlices
        fn validInplace(res: Tensor, anyargs: anytype) bool { //TODO: improve
            if (Tensor.dims.len == 0) return true;
            const AnyArgs = @TypeOf(anyargs);
            inline for (@typeInfo(AnyArgs).Struct.fields) |field_anyargs| {
                const AnyArg = field_anyargs.type;
                if (is(AnyArg)) {
                    const anyarg = @field(anyargs, field_anyargs.name);
                    if (std.meta.eql(anyarg.vals, res.vals)) { // inplace deteted
                        if (AnyArg != Tensor) return false;
                        if (!std.meta.eql(anyarg, res)) return false;
                    }
                }
            }
            return true;
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
    const s = try S.init(&.{}, ally);
    defer s.deinit(ally);
    try expect(s.vals.len == 1);
    try expect(@TypeOf(s.size.vec) == @Vector(0, usize));
}

test "tensor at/set" {
    const ally = std.testing.allocator;
    const S = Type(f32).Dense(&.{ 0, 1 });
    const s = try S.init(&.{ 2, 3 }, ally);
    defer s.deinit(ally);
    s.set(&.{ 0, 2, 4 }, 6);
    try expect(s.at(&.{ 0, 2 }) == 6);
}

test "tensor f" {
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

    d = T.f(op.add, op.mul, .{ a, b.t(0, 1) }); //combined
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

    const b = a.clamp(1, 1, 2);
    try expect(b.vals.ptr == a.vals.ptr);
    try expect(b.offset == 2);
    try expect(b.size.vec[0] == 2);
    try expect(b.size.vec[1] == 2);
    try expect(b.at(&.{ 0, 0 }) == 2);
    try expect(b.at(&.{ 0, 1 }) == 3);
    try expect(b.at(&.{ 1, 0 }) == 5);
    try expect(b.at(&.{ 1, 1 }) == 6);

    const c = a.sub(0, 1);
    const V = @TypeOf(c);
    try expect(V.dims.len == 1);
    try expect(V.dims.ptr[0] == 1);
    try expect(c.vals.ptr == a.vals.ptr);
    try expect(c.offset == 1);
    try expect(c.size.vec[0] == 3);
    try expect(c.at(&.{ 0, 0 }) == 4);
    try expect(c.at(&.{ 0, 1 }) == 5);
    try expect(c.at(&.{ 0, 2 }) == 6);
}

test "tensor diag" {
    const op = @import("op.zig");
    const ally = std.testing.allocator;
    const T = Type(usize);
    const V = T.Dense(&.{0});
    const M = T.Dense(&.{ 0, 1 });
    const a = try V.init(&.{4}, ally);
    defer a.deinit(ally);
    for (0..a.size.at(0)) |i| {
        a.set(&.{i}, i + 1);
    }
    const b = try M.init(&.{ a.size.at(0), a.size.at(0) }, ally);
    defer b.deinit(ally);
    b.f(undefined, op.add, .{ a, a.t(0, 1) });
    a.f(undefined, op.mul, .{ a, 2 });
    const c: V = b.diag(0, 1);

    try expect(Type(bool).f(op.@"and", op.eq, .{ a, c }));
}
