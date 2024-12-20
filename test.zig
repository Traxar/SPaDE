const std = @import("std");
const expect = std.testing.expect;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;

inline fn _alloc(Element: type, comptime length: usize) []Element {
    var mem: [length]Element = undefined;
    return mem[0..];
}

fn index(slice: []usize) void {
    for (slice, 0..) |*s, i| {
        s.* = i;
    }
}

test {
    const mem = _alloc(usize, 3);
    try expect(mem.len == 3);
    index(mem);
    try expect(mem[0] == 0);
    try expect(mem[1] == 1);
    try expect(mem[2] == 2);
}
