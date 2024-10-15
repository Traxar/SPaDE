const std = @import("std");
const expect = std.testing.expect;
const Allocator = std.mem.Allocator;
const assert = std.debug.assert;
const util = @import("util.zig");
const MultiSlice = util.MultiSlice;
const common = @import("common.zig");
const Majority = common.Majority;

/// return type of dense matrix
/// with field type 'Element'
/// and majority 'maj'
pub fn Type(comptime Elem: type, comptime majority: Majority) type {
    return struct {
        pub const Element = Elem;
        pub const major = majority;
        pub const minor = major.other();
        const Dense = @This();
        const DenseT = Type(Element, major.other());

        val: MultiSlice(Element) = .{ .ptr = undefined, .len = 0 },
        rows: usize = 0,
        cols: usize = 0,

        fn sizeMajor(a: Dense) usize {
            return switch (major) {
                .row => a.cols,
                .col => a.rows,
            };
        }

        fn sizeMinor(a: Dense) usize {
            return switch (major) {
                .row => a.rows,
                .col => a.cols,
            };
        }

        fn SimdVal(a: Dense, comptime step: usize) MultiSlice(Element.SimdType(step)) {
            return @bitCast(a.val);
        }

        pub fn init(rows: usize, cols: usize, allocator: Allocator) !Dense {
            return Dense{
                .val = try MultiSlice(Element).init(rows * cols, allocator),
                .rows = rows,
                .cols = cols,
            };
        }

        pub fn deinit(a: Dense, allocator: Allocator) void {
            a.val.deinit(allocator);
        }

        pub fn ensureCapacity(a: *Dense, rows: usize, cols: usize, allocator: Allocator) !void {
            try a.val.ensureCapacity(rows * cols, allocator);
            a.rows = rows;
            a.cols = cols;
        }

        pub const Index = struct {
            row: usize,
            col: usize,
            ind: usize,

            pub fn prev(ind: *Index, comptime dir: Majority, comptime step: usize, a: Dense) bool {
                comptime { // asserts
                    if (step == 0) @compileError("step size must be > 0");
                    if (dir == major and step != 1) @compileError("step in minor direction must have size 1");
                }
                //check if prev exists
                const active = switch (dir) {
                    .row => &ind.row,
                    .col => &ind.col,
                };
                if (active.* <= step - 1) return false;
                //calc prev
                active.* -= step;
                ind.ind -= if (dir == major) a.sizeMajor() else step;
                return true;
            }
        };

        /// index at row i and column j
        fn index(a: Dense, i: usize, j: usize) Index {
            assert(i <= a.rows and j <= a.cols); // out of bounds
            return .{
                .row = i,
                .col = j,
                .ind = switch (major) {
                    .row => i * a.cols + j,
                    .col => j * a.rows + i,
                },
            };
        }

        /// element at row i and column j
        pub fn at(a: Dense, i: usize, j: usize) Element {
            assert(i < a.rows and j < a.cols); // out of bounds
            return a.val.at(a.index(i, j).ind);
        }

        /// set element at row i and column j to b
        pub fn set(a: Dense, i: usize, j: usize, b: Element) void {
            assert(i < a.rows and j < a.cols); // out of bounds
            a.val.set(a.index(i, j).ind, b);
        }

        /// cast to transpose using other majority
        pub fn t(a: Dense) DenseT {
            return .{
                .val = a.val,
                .rows = a.cols,
                .cols = a.rows,
            };
        }

        /// res <- a^T
        pub fn transpose(res: Dense, a: Dense) void {
            assert(res.rows == a.cols and res.cols == a.rows); // dimensions do not match
            const inplace = std.meta.eql(a.val, res.val);
            var iter = res.index(res.rows, res.cols);
            var iter_ = a.index(a.rows, a.cols);
            const iter_a = if (inplace) &iter else &iter_;
            while (iter.prev(major, 1, res)) {
                assert(iter_a.prev(minor, 1, a));
                var ind = iter;
                var ind_a = iter_a.*;
                while (ind.prev(minor, 1, res)) {
                    assert(ind_a.prev(major, 1, a));
                    if (inplace) {
                        const h = res.val.at(ind.ind);
                        res.val.set(ind.ind, a.val.at(ind_a.ind));
                        a.val.set(ind_a.ind, h);
                    } else {
                        res.val.set(ind.ind, a.val.at(ind_a.ind));
                    }
                }
            }
        }

        ///set arguments given as matrix
        fn argsSet(comptime step: usize, ind: usize, args: anytype, a: *common.ArgsType(step, @TypeOf(args))) void {
            const info_args = @typeInfo(@TypeOf(args)).Struct;
            inline for (info_args.fields) |arg| {
                if (common.isElement(arg.type)) continue;
                @field(a, arg.name) = @field(args, arg.name).SimdVal(step).at(ind);
            }
        }

        fn ErrorSet(comptime op: anytype, comptime Args: type) ?type {
            return util.ErrorSet(op, common.ArgsType(1, Args));
        }

        ///res <- op(args)
        pub fn ew(res: Dense, comptime op: anytype, args: anytype) if (ErrorSet(op, @TypeOf(args))) |E| E!void else void {
            const Args = @TypeOf(args);
            if (common.argsMajority(Args)) |maj| {
                if (maj != major) @compileError("majority does not match");
                const dim = common.argsDimensions(args);
                assert(dim.rows == res.rows and dim.cols == res.cols);
                const simd = common.argsSimdSize(Args);
                const steps = if (simd > 1) .{ simd, 1 } else .{1};
                var i = res.cols * res.rows;
                inline for (steps) |step| {
                    var a = common.argsPrep(step, args);
                    while (i > step - 1) {
                        i -= step;
                        argsSet(step, i, args, &a);
                        const r_ = @call(.auto, op, a);
                        const r = if (ErrorSet(op, @TypeOf(args))) |_| try r_ else r_;
                        res.SimdVal(step).set(i, r);
                    }
                }
            } else {
                const r_ = @call(.auto, op, args);
                const r = if (ErrorSet(op, @TypeOf(args))) |_| try r_ else r_;
                res.val.fill(0, res.cols * res.rows, r);
            }
        }
    };
}

test {
    const F = @import("float.zig").Type(f32);
    const MP = MultiSlice(F);
    const FSimd = F.SimdType(null);
    const MPSimd = MultiSlice(FSimd);
    const mp: MP = undefined;
    const mp_simd: MPSimd = @bitCast(mp);
    _ = mp_simd;
}

test "matrix transpose" {
    const ally = std.testing.allocator;
    const F = @import("float.zig").Type(f64);
    const M = Type(F, .row);

    const n = 3;
    const m = 4;
    const a = try M.init(n, m, ally);
    defer a.deinit(ally);
    for (0..n) |i| {
        for (0..m) |j| {
            a.set(i, j, F.from(@intCast(i), j + 1));
        }
    }

    //transpose
    const aT = a.t();
    for (0..m) |i| {
        for (0..n) |j| {
            try expect(aT.at(i, j).eq(F.from(@intCast(j), i + 1)));
        }
    }
    var b = M{};
    defer b.deinit(ally);
    try b.ensureCapacity(m, n, ally);
    b.transpose(a);
    for (0..m) |i| {
        for (0..n) |j| {
            try expect(b.at(i, j).eq(aT.at(i, j)));
        }
    }
}

test "matrix elementwise operations" {
    const ally = std.testing.allocator;
    const F = @import("float.zig").Type(f64);
    const M = Type(F, .row);

    const n = 3;
    const m = 4;
    const a = try M.init(n, m, ally);
    defer a.deinit(ally);
    for (0..n) |i| {
        for (0..m) |j| {
            a.set(i, j, F.from(@intCast(i), j + 1));
        }
    }

    const b = try M.init(n, m, ally);
    defer b.deinit(ally);
    try b.ew(F.div, .{ F.one, F.one });
    b.ew(F.mul, .{ a, b });
    b.ew(F.add, .{ b, F.one });
    for (0..n) |i| {
        for (0..m) |j| {
            try expect(b.at(i, j).eq(F.from(@intCast(i), j + 1).add(F.one)));
        }
    }
}
