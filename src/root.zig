const std = @import("std");
pub const Tensor = @import("tensor.zig").Type;
pub const op = @import("op.zig");

test {
    std.testing.refAllDeclsRecursive(@This());
}
