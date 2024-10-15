const std = @import("std");
const expect = std.testing.expect;

inline fn add(a: anytype, b: @TypeOf(a)) @TypeOf(a) {
    return a + b;
}

test {
    const Ts = .{ usize, f32 };
    inline for (Ts) |T| {
        const a: T = 3;
        const b: T = 2;
        const c = @call(.auto, add, .{ a, b });

        try expect(@TypeOf(c) == T);
        try expect(c == 5);
    }
}

test {
    try expect(@bitSizeOf(bool) == 1);
    try expect(@bitSizeOf(u1) == 1);
}

const A = struct {
    pub const B: bool = false;
    c: u8,
    pub fn f(x: anytype) @TypeOf(x) {
        return x;
    }
    pub fn p(x: anytype) @TypeOf(x.*) {
        return x.*;
    }
};

test {
    try expect(@hasDecl(A, "B"));
    try expect(!@hasDecl(A, "c"));
    try expect(@hasDecl(A, "f"));
    try expect(!@hasField(A, "B"));
    try expect(@hasField(A, "c"));
    try expect(!@hasField(A, "f"));

    var a: A = .{ .c = 1 };
    try expect(@field(A, "f") == A.f);
    a = @call(.auto, @field(A, "f"), .{a}); // a is interpreted as field
    a = a.p(); // a is interpreted as pointer
    //a = a.f(a);
    //try expect(false);
}

test {
    const a = .{@as(usize, 3)};
    const info = @typeInfo(@TypeOf(a)).Struct;
    const field = info.fields[0];
    try expect(field.default_value != null);
    //const b: usize = field.default_value.?.*;
    //try expect(b == 3);

}
