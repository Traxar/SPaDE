pub const Tensor = @import("tensor.zig").Type;
pub const op = @import("op.zig");

test {
    _ = Tensor;
    _ = op;
}
