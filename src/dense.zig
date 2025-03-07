const std = @import("std");
const expect = std.testing.expect;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const util = @import("util.zig");
const Dims = @import("dims.zig").Type;
const Layout = @import("layout.zig").Type;
const Coords = @import("coords.zig").Type;
const Arg = @import("args.zig").Type;
const simd = @import("simd.zig");

/// Returns `true` if `T` is a dense tensor.
pub inline fn is(T: type) bool {
    comptime {
        if (@typeInfo(T) != .@"struct") return false;
        if (!@hasDecl(T, "Element") or @TypeOf(T.Element) != type) return false;
        if (!@hasDecl(T, "dims") or @TypeOf(T.dims) != Dims) return false;
        return T == Type(T.Element, T.dims);
    }
}

pub fn Type(Element_: type, comptime dims_: Dims) type {
    return packed struct {
        const Dense = @This();
        pub const Element = Element_;
        pub const dims = dims_;

        const Data = util.MultiSlice(Element, null);

        layout: Layout(dims),
        vals: Data,

        /// Returns a newly allocated tensor with size `size` and `undefined` values.
        /// - `size` is a multiindex. All entries at an index not in `dims` will be ignored.
        /// - Free the result by using `deinit`.
        ///
        /// (ex.: `Type(f32).Dense(&.{0, 1}).init(&.{2, 3})` tries to allocate a 2x3 matrix)
        pub fn init(size_: []const usize, allocator: Allocator) !Dense {
            const layout = Layout(dims).from(size_);
            return .{
                .layout = layout,
                .vals = try Data.init(layout.n(), allocator),
            };
        }

        /// Sets the size of tensor `tensor` to the new size `size`, while reusing the already allocated memory if possible.
        /// - `size` is a multiindex. All entries at an index not in `dims` will be ignored.
        /// - invalidates data
        pub fn reinit(tensor: *Dense, size_: []const usize, allocator: Allocator) !void {
            const layout = Layout(dims).from(size_);
            try tensor.vals.reinit(layout.n(), allocator);
            tensor.layout = layout;
        }

        /// Frees memory allocated by tensor `tensor`.
        pub fn deinit(tensor: Dense, allocator: Allocator) void {
            tensor.vals.deinit(allocator);
        }

        /// Returns the size of tensor `tensor` in dimension `d`
        pub fn size(tensor: Dense, comptime d: usize) usize {
            return tensor.layout.size.at(d);
        }

        /// Returns value of tensor `tensor` at coordinates `coords`.
        /// - `coord` is a multiindex. All entries at an index not in `dims` will be ignored.
        pub fn at(tensor: Dense, coords: []const usize) Element {
            return tensor.vals.at(tensor.layout.index(coords));
        }

        /// Sets tensor `tensor` at coordinates `coords` to the new value `value`.
        /// - `coord` is a multiindex. All entries at an index not in `dims` will be ignored.
        pub fn set(tensor: Dense, coords: []const usize, value: Element) void {
            tensor.vals.set(tensor.layout.index(coords), value);
        }

        /// Lazily swap dimensions `i` and `j`.
        /// - this has no cost
        pub fn t(tensor: Dense, comptime i: usize, comptime j: usize) Type(Element, dims.swap(i, j)) {
            return @bitCast(tensor);
        }

        /// Lazily restrict the coordinate of dimension `d` to size `size` starting at `start`.
        pub fn clamp(tensor: Dense, comptime d: usize, start: usize, _size: usize) Dense {
            return .{
                .layout = tensor.layout.clamp(d, start, _size),
                .vals = tensor.vals,
            };
        }

        /// Lazily fix the coordinate in dimension `d` to `coord`.
        /// - This returns a tensor of lower order.
        pub fn sub(tensor: Dense, comptime d: usize, coord: usize) Type(Element, dims.sub(Dims.from(&.{d}))) {
            return .{
                .layout = tensor.layout.sub(d, coord),
                .vals = tensor.vals,
            };
        }

        /// Lazily take the diagonal of dimensions `i` and `j`.
        /// - The result is a tensor without dimension `j`.
        pub fn diag(tensor: Dense, comptime i: usize, comptime j: usize) Type(Element, dims.sub(Dims.from(&.{j}))) {
            return .{
                .layout = tensor.layout.diag(i, j),
                .vals = tensor.vals,
            };
        }

        /// Sets `res` to the result of the tensor operation:
        /// - `args` is an anonymous struct with the arguments of the operation.
        /// - `ew` is a function which is applied *elementwise*. It takes `args`, with each tensor swapped out for one of its elements,
        ///   as input and returns `Element` or `!Element`.
        /// - `red` is a function used to *reduce* the results of `ew` to fit the `res.dims`. It takes two `Element`s as input and returns one `Element`.
        pub fn f(res: Dense, comptime red: anytype, comptime ew: anytype, args: anytype) Arg(@TypeOf(args)).ErrorWrap(ew, void) {
            const A = Arg(@TypeOf(args));
            return res.fInternal(red, ew, args, dims.unite(A.dims));
        }

        /// res <- red(ew(args))
        /// with iteration order given by `dims_order`
        fn fInternal(res: Dense, comptime red: anytype, comptime ew: anytype, args: anytype, comptime dims_order: Dims) Arg(@TypeOf(args)).ErrorWrap(ew, void) {
            const A = Arg(@TypeOf(args));
            const dims_total = dims.unite(A.dims);
            assert(dims_total.equal(dims_order)); //dims_order must match dimensions of the tensor operation
            const dims_fill = dims_order.sub(A.dims);
            const dims_red = dims_order.sub(dims);
            const dims_calc = dims_order.sub(dims_fill).sub(dims_red);
            assert(res.allValidInplace(args, dims_calc)); //inplace operation not valid
            const bounds = res.collectBounds(args);
            var a = A.init(args);
            var coord_iter: Coords(dims_total) = undefined;
            coord_iter.reset(dims_calc);
            while (true) {
                coord_iter.reset(dims_red);
                a.set(args, &coord_iter.arr);
                const res_red_err = a.call(ew);
                var res_red: Element = if (util.ErrorSet(@TypeOf(res_red_err))) |_| try res_red_err else res_red_err;
                while (dims_red.len > 0 and coord_iter.next(bounds, dims_red)) {
                    a.set(args, &coord_iter.arr);
                    const res_ew_err = a.call(ew);
                    const res_ew: Element = if (util.ErrorSet(@TypeOf(res_ew_err))) |_| try res_ew_err else res_ew_err;
                    res_red = red(res_red, res_ew); //`red` must be a reduction
                }
                coord_iter.reset(dims_fill);
                while (true) {
                    res.set(&coord_iter.arr, res_red);
                    if (!coord_iter.next(bounds, dims_fill)) break;
                }
                if (!coord_iter.next(bounds, dims_calc)) break;
            }
        }

        /// union of all dimensions of `@This()` and `Args`
        inline fn CoordsType(Args: type) type {
            comptime {
                return Coords(dims.unite(Arg(Args).dims));
            }
        }

        /// total bounds of tensor `res` and arguments `args`
        /// asserts matching bounds
        fn collectBounds(res: Dense, args: anytype) CoordsType(@TypeOf(args)) {
            var bounds = CoordsType(@TypeOf(args)).zero;
            bounds.collectBounds(res);
            bounds.collectBoundsMany(args);
            return bounds;
        }

        fn allValidInplace(res: Dense, args: anytype, comptime dims_calc: Dims) bool {
            if (dims_calc.len == 0) return true;
            inline for (@typeInfo(@TypeOf(args)).@"struct".fields) |field_args| {
                if (!res.validInplace(@field(args, field_args.name), dims_calc)) return false;
            }
            return true;
        }

        fn validInplace(res: Dense, arg: anytype, comptime dims_calc: Dims) bool {
            assert(dims_calc.len != 0);
            const A = @TypeOf(arg);
            // quick checks
            if (!is(A)) return true;
            if (!std.meta.eql(res.vals, arg.vals)) return true;
            if (A == Dense and std.meta.eql(res, arg)) return true;
            // actual check
            return res.layout.validInplace(arg.layout, dims_calc);
        }
    };
}

test "dense type" {
    const ally = std.testing.allocator;
    const S = Type(f32, Dims.from(&.{ 1, 2 }));
    const s = try S.init(&.{ 3, 4, 5 }, ally); //unused dimension 0 is ignored
    defer s.deinit(ally);
    try expect(s.vals.len == 20);
}

test "dense type 0D" {
    const ally = std.testing.allocator;
    const S = Type(f32, Dims.from(&.{}));
    const s = try S.init(&.{}, ally);
    defer s.deinit(ally);
    try expect(s.vals.len == 1);
}

test "dense at/set" {
    const ally = std.testing.allocator;
    const S = Type(f32, Dims.from(&.{ 0, 1 }));
    const s = try S.init(&.{ 2, 3 }, ally);
    defer s.deinit(ally);
    s.set(&.{ 0, 2, 4 }, 6);
    try expect(s.at(&.{ 0, 2 }) == 6);
}

test "dense f" {
    const op = @import("op.zig");
    const ally = std.testing.allocator;
    const T = @import("tensor.zig").Type(f32);
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

test "dense sub/fix" {
    const ally = std.testing.allocator;
    const M = Type(f32, Dims.from(&.{ 0, 1 }));
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
    try expect(b.at(&.{ 0, 0 }) == 2);
    try expect(b.at(&.{ 0, 1 }) == 3);
    try expect(b.at(&.{ 1, 0 }) == 5);
    try expect(b.at(&.{ 1, 1 }) == 6);

    const c = a.sub(0, 1);
    const V = @TypeOf(c);
    try expect(V.dims.len == 1);
    try expect(V.dims.ptr[0] == 1);
    try expect(c.vals.ptr == a.vals.ptr);
    try expect(c.at(&.{ 0, 0 }) == 4);
    try expect(c.at(&.{ 0, 1 }) == 5);
    try expect(c.at(&.{ 0, 2 }) == 6);
}

test "dense diag" {
    const op = @import("op.zig");
    const ally = std.testing.allocator;
    const V = Type(usize, Dims.from(&.{0}));
    const M = Type(usize, Dims.from(&.{ 0, 1 }));
    const a = try V.init(&.{4}, ally);
    defer a.deinit(ally);
    for (0..a.size(0)) |i| {
        a.set(&.{i}, i + 1);
    }
    const b = try M.init(&.{ a.size(0), a.size(0) }, ally);
    defer b.deinit(ally);
    b.f(undefined, op.add, .{ a, a.t(0, 1) });
    a.f(undefined, op.mul, .{ a, 2 });
    const c: V = b.diag(0, 1);
    try expect(@import("tensor.zig").Type(bool).f(op.@"and", op.eq, .{ a, c }));
}
