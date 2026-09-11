use std::{env, error::Error, fs, path::PathBuf};

const GFX_WIDTH: usize = 128;
const GFX_HEIGHT: usize = 128;

fn main() -> Result<(), Box<dyn Error>> {
    println!("cargo:rustc-check-cfg=cfg(test)");
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let source_path = manifest_dir.join("../../../src/dodge/game/dodge.p8");
    println!("cargo:rerun-if-changed={}", source_path.display());

    let source = fs::read_to_string(&source_path)?;
    let marker = "__gfx__\n";
    let gfx_source = source
        .split_once(marker)
        .ok_or("cartridge has no __gfx__ section")?
        .1;
    let mut pixels = Vec::with_capacity(GFX_WIDTH * GFX_HEIGHT);
    for line in gfx_source.lines() {
        if line.starts_with("__") {
            break;
        }
        if line.is_empty() {
            continue;
        }
        if line.len() != GFX_WIDTH {
            return Err(format!("gfx row has {} characters", line.len()).into());
        }
        for byte in line.bytes() {
            pixels.push(hex_digit(byte).ok_or("gfx row contains a non-hex digit")?);
        }
    }
    if pixels.len() > GFX_WIDTH * GFX_HEIGHT {
        return Err("gfx section has more than 128 rows".into());
    }
    pixels.resize(GFX_WIDTH * GFX_HEIGHT, 0);

    let mut generated = String::from("pub static GFX_INDICES: [u8; 16384] = [\n");
    for chunk in pixels.chunks(32) {
        generated.push_str("    ");
        for (index, pixel) in chunk.iter().enumerate() {
            if index != 0 {
                generated.push_str(", ");
            }
            generated.push_str(&pixel.to_string());
        }
        generated.push_str(",\n");
    }
    generated.push_str("];\n");
    let output_path =
        PathBuf::from(env::var_os("OUT_DIR").ok_or("OUT_DIR is unset")?).join("gfx_indices.rs");
    fs::write(output_path, generated)?;
    Ok(())
}

fn hex_digit(byte: u8) -> Option<u8> {
    match byte {
        b'0'..=b'9' => Some(byte - b'0'),
        b'a'..=b'f' => Some(byte - b'a' + 10),
        b'A'..=b'F' => Some(byte - b'A' + 10),
        _ => None,
    }
}
