const std = @import("std");
const expect = std.testing.expect;
const assert = std.debug.assert;

/// return the non pointer type
pub fn Deref(A: type) type {
    const info = @typeInfo(A);
    return if (info == .pointer) info.pointer.child else A;
}

test {
    try expect(Deref(usize) == usize);
    try expect(Deref(*usize) == usize);
}

/// ReturnType of the function `f` given arguments of type `Args`.
pub fn ReturnType(comptime f: anytype, comptime Args: type) type {
    const args: Args = undefined;
    return @TypeOf(@call(.auto, f, args));
}

/// ErrorSet of the type `T`.
/// `null` if T contains no error.
pub fn ErrorSet(T: type) ?type {
    return switch (@typeInfo(T)) {
        .error_union => |error_union| error_union.error_set,
        else => null,
    };
}

/// stack allocation
pub inline fn stackAlloc(Element: type, comptime length: usize) []Element {
    var mem: [length]Element = undefined;
    return mem[0..];
}

pub inline fn len(T: type) usize {
    return switch (@typeInfo(T)) {
        .array => |a| a.len,
        .vector => |v| v.len,
        else => unreachable,
    };
}

pub fn Child(T: type) type {
    return switch (@typeInfo(T)) {
        .array => |a| a.child,
        .vector => |v| v.child,
        else => unreachable,
    };
}
