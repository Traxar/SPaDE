const std = @import("std");
const expect = std.testing.expect;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const tensor = @import("src/tensor.zig");

const T = tensor.Type(f32); //tensor base type, supports arbitrary element types
const V = T.Dense(&.{0}); //vector
const Mcol = T.Dense(&.{ 0, 1 }); //column major matrix
const Mrow = T.Dense(&.{ 1, 0 }); //row major matrix

// custom functions for tensor operations:
fn id(a: anytype) @TypeOf(a) {
    return a;
}

fn add(a: anytype, b: anytype) @TypeOf(a + b) {
    return a + b;
}

fn mul(a: anytype, b: anytype) @TypeOf(a * b) {
    return a * b;
}

/// matrix multiplication
fn matMut(res: anytype, a: anytype, b: anytype) void {
    const Res = @TypeOf(res);
    const A = @TypeOf(a);
    const B = @TypeOf(b);
    if (Res != Mcol and Res != Mrow) @compileError("res must be of type Mcol or Mrow");
    if (A != Mcol and A != Mrow) @compileError("res must be of type Mcol or Mrow");
    if (B != Mcol and B != Mrow) @compileError("res must be of type Mcol or Mrow");
    res.f(add, mul, .{ a.t(1, 2), b.t(0, 2) });
}

test {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer if (gpa.deinit() == .leak) @panic("MEMORY LEAKED");
    const allocator = gpa.allocator();

    const a = try Mcol.init(&.{ 2, 3 }, allocator); //allocate 2 x 3 matrix
    defer a.deinit(allocator);
    a.set(&.{ 0, 0 }, 1); //setting elements 1 by 1
    a.set(&.{ 0, 1 }, 2);
    a.set(&.{ 0, 2 }, 3);
    a.set(&.{ 1, 0 }, -1);
    a.set(&.{ 1, 1 }, -2);
    a.set(&.{ 1, 2 }, -3);
    //     / 1  2  3 \
    // a = |         |
    //     \-1 -2 -3 /

    const b = try Mcol.init(&.{ 4, 3 }, allocator); //allocate 4 x 3 matrix
    defer b.deinit(allocator);
    b.f(undefined, id, .{0}); //fill with zeros
    b.clamp(0, 2, 2).f(undefined, id, .{1}); //fill the lower half of the matrix
    assert(@TypeOf(b.clamp(0, 2, 2)) == Mcol); //.clamp gives a sub tensor of same type
    b.sub(1, 0).f(undefined, id, .{2}); //fill the first column of the matrix
    assert(@TypeOf(b.sub(1, 0)) == V); //.sub gives a sub tensor of lower dimension
    //     / 2  0  0 \
    // b = | 2  0  0 |
    //     | 2  1  1 |
    //     \ 2  1  1 /

    const c = try Mcol.init(&.{ 2, 4 }, allocator); //allocate 4 x 3 matrix
    defer c.deinit(allocator);
    matMut(c, a, b.t(0, 1)); // c = a * b^T
    assert(@TypeOf(b.t(0, 1)) == Mrow); //.t swaps 2 dimension

    const stdout_file = std.io.getStdOut().writer();
    var bw = std.io.bufferedWriter(stdout_file);
    const stdout = bw.writer();
    try stdout.print("a * b^T =\n", .{});
    for (0..2) |i| {
        for (0..4) |j| {
            try stdout.print("{} ", .{c.at(&.{ i, j })});
        }
        try stdout.print("\n", .{});
    }
    try bw.flush();
}

test {
    const a: @Vector(4, usize) = .{ 2, 3, 4, 5 };
    try expect(@reduce(.And, @divFloor(a, a) != a));
}

test {
    _ = @max(@as(u32, 10), @as(i32, 10));
}
