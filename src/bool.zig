pub fn Type(comptime size: usize) type {
    return if (size == 1) bool else @Vector(size, bool);
}

/// b must be a bool or vector of bools
pub inline fn all(b: anytype) bool {
    return if (@TypeOf(b) == bool) b else @reduce(.And, b);
}
