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

        vals: MultiSlice(Element) = .{ .ptr = undefined, .len = 0 },
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

        fn valsSimd(a: Dense, comptime step: usize) MultiSlice(Element.SimdType(step)) {
            return @bitCast(a.vals);
        }

        pub fn init(rows: usize, cols: usize, allocator: Allocator) !Dense {
            return Dense{
                .vals = try MultiSlice(Element).init(rows * cols, allocator),
                .rows = rows,
                .cols = cols,
            };
        }

        pub fn deinit(a: Dense, allocator: Allocator) void {
            a.vals.deinit(allocator);
        }

        pub fn ensureCapacity(a: *Dense, rows: usize, cols: usize, allocator: Allocator) !void {
            try a.vals.ensureCapacity(rows * cols, allocator);
            a.rows = rows;
            a.cols = cols;
        }

        pub const Index = struct {
            row: usize,
            col: usize,
            ind: usize,

            pub fn prev(ind: *Index, comptime dir: Majority, comptime step: usize, a: Dense) bool {
                assert(step > 0);
                if (dir == major) assert(step == 1);
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
            assert(i <= a.rows);
            assert(j <= a.cols);
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
            assert(i < a.rows);
            assert(j < a.cols);
            return a.vals.at(a.index(i, j).ind);
        }

        /// set element at row i and column j to b
        pub fn set(a: Dense, i: usize, j: usize, b: Element) void {
            assert(i < a.rows);
            assert(j < a.cols);
            a.vals.set(a.index(i, j).ind, b);
        }

        /// cast to transpose using other majority
        pub fn t(a: Dense) DenseT {
            return .{
                .vals = a.vals,
                .rows = a.cols,
                .cols = a.rows,
            };
        }

        /// res <- a^T
        pub fn transpose(res: Dense, a: Dense) void {
            assert(res.rows == a.cols);
            assert(res.cols == a.rows);
            const inplace = std.meta.eql(a.vals, res.vals);
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
                        const h = res.vals.at(ind.ind);
                        res.vals.set(ind.ind, a.vals.at(ind_a.ind));
                        a.vals.set(ind_a.ind, h);
                    } else {
                        res.vals.set(ind.ind, a.vals.at(ind_a.ind));
                    }
                }
            }
        }

        /// set arguments given as matrix
        fn argsSet(comptime step: usize, ind: usize, args: anytype, a: *common.ArgsType(step, @TypeOf(args))) void {
            const info_args = @typeInfo(@TypeOf(args)).Struct;
            inline for (info_args.fields) |arg| {
                if (arg.type.Element == arg.type) continue;
                @field(a, arg.name) = @field(args, arg.name).valsSimd(step).at(ind);
            }
        }

        fn ErrorSet(comptime op: anytype, comptime Args: type) ?type {
            return util.ErrorSet(op, common.ArgsType(1, Args));
        }

        ///res <- op(args)
        pub fn ew(res: Dense, comptime op: anytype, args: anytype) if (ErrorSet(op, @TypeOf(args))) |E| E!void else void {
            const Args = @TypeOf(args);
            if (common.argsMajority(Args)) |maj| {
                assert(maj == major);
                const dim = common.argsDimensions(args);
                assert(dim.rows == res.rows);
                assert(dim.cols == res.cols);
                var i = dim.cols * dim.rows;
                inline for (.{ common.argsSimdSize(Args), 1 }) |step| {
                    var args_step = common.argsPrep(step, args);
                    while (i >= step) {
                        i -= step;
                        argsSet(step, i, args, &args_step);
                        const res_step_err = @call(.auto, op, args_step);
                        const res_step = if (ErrorSet(op, @TypeOf(args))) |_| try res_step_err else res_step_err;
                        res.valsSimd(step).set(i, res_step);
                    }
                }
            } else {
                const res_all_err = @call(.auto, op, args);
                const res_all = if (ErrorSet(op, @TypeOf(args))) |_| try res_all_err else res_all_err;
                res.vals.fill(0, res.cols * res.rows, res_all);
            }
        }

        ///res <- op_red(op_ew(args))
        pub fn red(comptime op_red: anytype, comptime op_ew: anytype, args: anytype) if (ErrorSet(op_ew, @TypeOf(args))) |E| E!Element else Element {
            const Args = @TypeOf(args);
            assert(common.argsMajority(Args) == major);
            const dim = common.argsDimensions(args);
            var i = dim.cols * dim.rows;
            assert(i > 0);
            var res: ?Element = null;
            inline for (.{ common.argsSimdSize(Args), 1 }) |step| blk: {
                var args_step = common.argsPrep(step, args);
                if (i < step) break :blk; //continue
                i -= step;
                argsSet(step, i, args, &args_step);
                const res_step_err = @call(.auto, op_ew, args_step);
                var res_step = if (ErrorSet(op_ew, @TypeOf(args))) |_| try res_step_err else res_step_err;
                while (i >= step) {
                    i -= step;
                    argsSet(step, i, args, &args_step);
                    const res_ew_err = @call(.auto, op_ew, args_step);
                    const res_ew = if (ErrorSet(op_ew, @TypeOf(args))) |_| try res_ew_err else res_ew_err;
                    res_step = op_red(res_ew, res_step);
                }
                const p = Element.simdReduce(res_step, op_red);
                res = if (res) |r| op_red(p, r) else p;
            }
            return res.?;
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
    const Float = @import("float.zig");
    const F = Float.Type(f64);
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

test "matrix reduce" {
    const ally = std.testing.allocator;
    const Float = @import("float.zig");
    const F = Float.Type(f64);
    const M = Type(F, .row);

    const n = 3;
    const m = 4;
    const a = try M.init(n, m, ally);
    defer a.deinit(ally);
    for (0..n) |i| {
        for (0..m) |j| {
            a.set(i, j, F.one);
        }
    }
    const b = try M.red(F.add, F.div, .{ a, F.from(2, 1) });
    try expect(b.eq(F.from(n * m, 2)));
}
