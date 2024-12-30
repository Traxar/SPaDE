# SPaDE

a library for **SP**arse **a**nd **DE**nse tensor operations.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Dependencies](#dependencies)

## Installation

in your zig-project run:

  zig fetch --save https://github.com/Traxar/SPaDE/archive/refs/tags/v0.0.1.tar.gz

Then add `spade` as an import to your root modules in `build.zig`:

```zig
fn build(b: *std.Build) void {
    // set build options ...

    const spade = b.dependency("spade", .{});

    // define exe ...

    exe.root_module.addImport("spade", spade.module("spade"));
}
```

## Usage

TODO: create simple usage sample

## Dependencies

- `zig 0.13.0`
