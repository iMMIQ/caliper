//! Minimal streaming ONNX metadata reader used for preflight input-shape checks.
//!
//! ONNX files are protobuf messages. This reader walks only ModelProto.graph,
//! GraphProto.input and initializer names, seeking over tensor payloads so a
//! multi-gigabyte model is not loaded into memory.

use anyhow::{bail, Context, Result};
use std::collections::HashSet;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;

#[derive(Debug, PartialEq)]
enum Dimension {
    Value(i64),
    Parameter(String),
    Unknown,
}

#[derive(Debug, PartialEq)]
struct TensorInput {
    name: String,
    shape: Option<Vec<Dimension>>,
}

impl TensorInput {
    fn is_dynamic(&self) -> bool {
        self.shape.as_ref().is_none_or(|dimensions| {
            dimensions
                .iter()
                .any(|dimension| !matches!(dimension, Dimension::Value(value) if *value > 0))
        })
    }

    fn display(&self) -> String {
        let dimensions = self.shape.as_ref().map_or_else(
            || "?".to_string(),
            |dimensions| {
                dimensions
                    .iter()
                    .map(|dimension| match dimension {
                        Dimension::Value(value) => value.to_string(),
                        Dimension::Parameter(value) if !value.is_empty() => value.clone(),
                        Dimension::Parameter(_) | Dimension::Unknown => "?".to_string(),
                    })
                    .collect::<Vec<_>>()
                    .join(",")
            },
        );
        format!("{}:[{}]", self.name, dimensions)
    }
}

/// Reject dynamic runtime inputs not covered by ATC's `--input_shape` value.
pub(crate) fn validate_input_shapes(path: &Path, input_shape: Option<&str>) -> Result<()> {
    let dynamic_inputs: Vec<_> = read_runtime_inputs(path)?
        .into_iter()
        .filter(TensorInput::is_dynamic)
        .collect();
    if dynamic_inputs.is_empty() {
        return Ok(());
    }

    let provided = input_shape_names(input_shape.unwrap_or_default());
    let missing: Vec<_> = dynamic_inputs
        .iter()
        .filter(|input| !provided.contains(input.name.as_str()))
        .map(TensorInput::display)
        .collect();
    if missing.is_empty() {
        return Ok(());
    }

    bail!(
        "ONNX 动态输入缺少静态 shape: {}。请通过 --input-shape（HTTP API 使用 JobSpec.input_shape）为每个动态输入指定 shape；已在调用 ATC 前退出",
        missing.join("; ")
    )
}

fn input_shape_names(input_shape: &str) -> HashSet<&str> {
    input_shape
        .split(';')
        .filter_map(|item| item.trim().split_once(':'))
        .filter_map(|(name, dimensions)| {
            let name = name.trim();
            (!name.is_empty() && !dimensions.trim().is_empty()).then_some(name)
        })
        .collect()
}

fn read_runtime_inputs(path: &Path) -> Result<Vec<TensorInput>> {
    let mut file =
        File::open(path).with_context(|| format!("打开 ONNX 失败: {}", path.display()))?;
    let file_end = file
        .metadata()
        .with_context(|| format!("读取 ONNX 元数据失败: {}", path.display()))?
        .len();

    while let Some((field, wire_type)) = read_key(&mut file, file_end)? {
        if field == 7 && wire_type == 2 {
            let graph_end = read_message_end(&mut file, file_end)?;
            return parse_graph(&mut file, graph_end)
                .with_context(|| format!("解析 ONNX graph 失败: {}", path.display()));
        }
        skip_field(&mut file, file_end, wire_type)?;
    }
    bail!("ONNX 缺少 graph: {}", path.display())
}

fn parse_graph(file: &mut File, end: u64) -> Result<Vec<TensorInput>> {
    let mut inputs = Vec::new();
    let mut initializers = HashSet::new();
    while let Some((field, wire_type)) = read_key(file, end)? {
        match (field, wire_type) {
            (5, 2) => {
                let message_end = read_message_end(file, end)?;
                if let Some(name) = parse_named_message(file, message_end, 8)? {
                    initializers.insert(name);
                }
                seek_to(file, message_end, end)?;
            }
            (11, 2) => {
                let message_end = read_message_end(file, end)?;
                if let Some(input) = parse_value_info(file, message_end)? {
                    inputs.push(input);
                }
                seek_to(file, message_end, end)?;
            }
            _ => skip_field(file, end, wire_type)?,
        }
    }
    inputs.retain(|input| !initializers.contains(&input.name));
    Ok(inputs)
}

fn parse_named_message(file: &mut File, end: u64, name_field: u32) -> Result<Option<String>> {
    let mut name = None;
    while let Some((field, wire_type)) = read_key(file, end)? {
        if field == name_field && wire_type == 2 {
            name = Some(read_string(file, end)?);
        } else {
            skip_field(file, end, wire_type)?;
        }
    }
    Ok(name)
}

fn parse_value_info(file: &mut File, end: u64) -> Result<Option<TensorInput>> {
    let mut name = None;
    let mut shape = None;
    while let Some((field, wire_type)) = read_key(file, end)? {
        match (field, wire_type) {
            (1, 2) => name = Some(read_string(file, end)?),
            (2, 2) => {
                let message_end = read_message_end(file, end)?;
                shape = parse_type(file, message_end)?;
                seek_to(file, message_end, end)?;
            }
            _ => skip_field(file, end, wire_type)?,
        }
    }
    Ok(name.map(|name| TensorInput { name, shape }))
}

fn parse_type(file: &mut File, end: u64) -> Result<Option<Vec<Dimension>>> {
    while let Some((field, wire_type)) = read_key(file, end)? {
        if field == 1 && wire_type == 2 {
            let message_end = read_message_end(file, end)?;
            return parse_tensor_type(file, message_end);
        }
        skip_field(file, end, wire_type)?;
    }
    Ok(None)
}

fn parse_tensor_type(file: &mut File, end: u64) -> Result<Option<Vec<Dimension>>> {
    while let Some((field, wire_type)) = read_key(file, end)? {
        if field == 2 && wire_type == 2 {
            let message_end = read_message_end(file, end)?;
            return parse_shape(file, message_end).map(Some);
        }
        skip_field(file, end, wire_type)?;
    }
    Ok(None)
}

fn parse_shape(file: &mut File, end: u64) -> Result<Vec<Dimension>> {
    let mut dimensions = Vec::new();
    while let Some((field, wire_type)) = read_key(file, end)? {
        if field == 1 && wire_type == 2 {
            let message_end = read_message_end(file, end)?;
            dimensions.push(parse_dimension(file, message_end)?);
            seek_to(file, message_end, end)?;
        } else {
            skip_field(file, end, wire_type)?;
        }
    }
    Ok(dimensions)
}

fn parse_dimension(file: &mut File, end: u64) -> Result<Dimension> {
    let mut dimension = Dimension::Unknown;
    while let Some((field, wire_type)) = read_key(file, end)? {
        match (field, wire_type) {
            (1, 0) => dimension = Dimension::Value(read_varint(file, end)? as i64),
            (2, 2) => dimension = Dimension::Parameter(read_string(file, end)?),
            _ => skip_field(file, end, wire_type)?,
        }
    }
    Ok(dimension)
}

fn read_key(file: &mut File, end: u64) -> Result<Option<(u32, u8)>> {
    if file.stream_position()? == end {
        return Ok(None);
    }
    let key = read_varint(file, end)?;
    let field = (key >> 3) as u32;
    let wire_type = (key & 0x07) as u8;
    if field == 0 {
        bail!("protobuf 字段编号为 0")
    }
    Ok(Some((field, wire_type)))
}

fn read_varint(file: &mut File, end: u64) -> Result<u64> {
    let mut value = 0u64;
    for shift in (0..70).step_by(7) {
        if file.stream_position()? >= end {
            bail!("protobuf varint 越界")
        }
        let mut byte = [0u8; 1];
        file.read_exact(&mut byte)?;
        value |= u64::from(byte[0] & 0x7f) << shift;
        if byte[0] & 0x80 == 0 {
            return Ok(value);
        }
    }
    bail!("protobuf varint 过长")
}

fn read_message_end(file: &mut File, parent_end: u64) -> Result<u64> {
    let length = read_varint(file, parent_end)?;
    let start = file.stream_position()?;
    let end = start
        .checked_add(length)
        .context("protobuf message 长度溢出")?;
    if end > parent_end {
        bail!("protobuf message 越界")
    }
    Ok(end)
}

fn read_string(file: &mut File, parent_end: u64) -> Result<String> {
    let end = read_message_end(file, parent_end)?;
    let start = file.stream_position()?;
    let length: usize = (end - start).try_into().context("protobuf 字符串过长")?;
    let mut bytes = vec![0; length];
    file.read_exact(&mut bytes)?;
    String::from_utf8(bytes).context("protobuf 字符串不是 UTF-8")
}

fn skip_field(file: &mut File, end: u64, wire_type: u8) -> Result<()> {
    match wire_type {
        0 => {
            read_varint(file, end)?;
        }
        1 => seek_forward(file, 8, end)?,
        2 => {
            let message_end = read_message_end(file, end)?;
            seek_to(file, message_end, end)?;
        }
        5 => seek_forward(file, 4, end)?,
        _ => bail!("不支持的 protobuf wire type: {wire_type}"),
    }
    Ok(())
}

fn seek_forward(file: &mut File, amount: u64, end: u64) -> Result<()> {
    let target = file
        .stream_position()?
        .checked_add(amount)
        .context("protobuf 字段长度溢出")?;
    seek_to(file, target, end)
}

fn seek_to(file: &mut File, target: u64, end: u64) -> Result<()> {
    if target > end {
        bail!("protobuf 字段越界")
    }
    file.seek(SeekFrom::Start(target))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_uncovered_dynamic_input() {
        let path = write_model(
            vec![value_info("input_ids", &[fixed_dim(1), param_dim("seq")])],
            vec![],
        );

        let error = validate_input_shapes(&path, None).unwrap_err().to_string();
        assert!(error.contains("input_ids:[1,seq]"));
        assert!(error.contains("调用 ATC 前退出"));
        validate_input_shapes(&path, Some("input_ids:1,64")).unwrap();

        std::fs::remove_file(path).unwrap();
    }

    #[test]
    fn accepts_static_inputs_and_ignores_initializer_inputs() {
        let path = write_model(
            vec![
                value_info("tokens", &[fixed_dim(1), fixed_dim(64)]),
                value_info("weight", &[param_dim("rows"), fixed_dim(64)]),
            ],
            vec![tensor("weight")],
        );

        validate_input_shapes(&path, None).unwrap();

        std::fs::remove_file(path).unwrap();
    }

    fn write_model(inputs: Vec<Vec<u8>>, initializers: Vec<Vec<u8>>) -> std::path::PathBuf {
        let mut graph = Vec::new();
        for initializer in initializers {
            bytes_field(&mut graph, 5, &initializer);
        }
        for input in inputs {
            bytes_field(&mut graph, 11, &input);
        }
        let mut model = Vec::new();
        bytes_field(&mut model, 7, &graph);
        let path = std::env::temp_dir().join(format!(
            "caliper-onnx-test-{}-{}.onnx",
            std::process::id(),
            uuid::Uuid::new_v4()
        ));
        std::fs::write(&path, model).unwrap();
        path
    }

    fn value_info(name: &str, dimensions: &[Vec<u8>]) -> Vec<u8> {
        let mut shape = Vec::new();
        for dimension in dimensions {
            bytes_field(&mut shape, 1, dimension);
        }
        let mut tensor_type = Vec::new();
        varint_field(&mut tensor_type, 1, 7);
        bytes_field(&mut tensor_type, 2, &shape);
        let mut value_type = Vec::new();
        bytes_field(&mut value_type, 1, &tensor_type);
        let mut value_info = Vec::new();
        bytes_field(&mut value_info, 1, name.as_bytes());
        bytes_field(&mut value_info, 2, &value_type);
        value_info
    }

    fn tensor(name: &str) -> Vec<u8> {
        let mut tensor = Vec::new();
        bytes_field(&mut tensor, 8, name.as_bytes());
        tensor
    }

    fn fixed_dim(value: u64) -> Vec<u8> {
        let mut dimension = Vec::new();
        varint_field(&mut dimension, 1, value);
        dimension
    }

    fn param_dim(value: &str) -> Vec<u8> {
        let mut dimension = Vec::new();
        bytes_field(&mut dimension, 2, value.as_bytes());
        dimension
    }

    fn varint_field(output: &mut Vec<u8>, field: u64, value: u64) {
        varint(output, field << 3);
        varint(output, value);
    }

    fn bytes_field(output: &mut Vec<u8>, field: u64, value: &[u8]) {
        varint(output, (field << 3) | 2);
        varint(output, value.len() as u64);
        output.extend_from_slice(value);
    }

    fn varint(output: &mut Vec<u8>, mut value: u64) {
        while value >= 0x80 {
            output.push((value as u8 & 0x7f) | 0x80);
            value >>= 7;
        }
        output.push(value as u8);
    }
}
