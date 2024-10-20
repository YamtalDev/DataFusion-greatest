// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements. See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership. The ASF licenses this
// file to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License. You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied. See the License for the
// specific language governing permissions and limitations
// under the License.

#![deny(clippy::clone_on_ref_ptr)]

//! [`ScalarUDFImpl`] definitions for the `greatest` function.

use crate::utils::make_scalar_function;
use arrow::array::*;
use arrow::datatypes::{
    DataType, Float32Type, Float64Type, Int16Type, Int32Type, Int64Type, Int8Type,
    UInt16Type, UInt32Type, UInt64Type, UInt8Type,
};
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::{
    type_coercion::binary::get_wider_type, ColumnarValue, Documentation, ScalarUDFImpl,
    Signature, Volatility,
};
use std::any::Any;
use std::sync::{Arc, OnceLock};

make_udf_expr_and_func!(
    Greatest,
    greatest,
    "Returns the greatest value among the arguments, skipping nulls.",
    greatest_udf
);

#[derive(Debug)]
pub struct Greatest {
    signature: Signature,
    aliases: Vec<String>,
}

impl Greatest {
    pub fn new() -> Self {
        Self {
            signature: Signature::variadic_any(Volatility::Immutable),
            aliases: vec![],
        }
    }
}

impl ScalarUDFImpl for Greatest {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn name(&self) -> &str {
        "greatest"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
        if arg_types.is_empty() {
            return Err(DataFusionError::Plan(
                "The 'greatest' function requires at least one argument".to_string(),
            ));
        }

        // Find the common supertype among the arguments
        let mut common_type = arg_types[0].clone();
        for arg_type in &arg_types[1..] {
            common_type = get_wider_type(&common_type, arg_type)?;
        }

        Ok(common_type)
    }

    fn invoke(&self, args: &[ColumnarValue]) -> Result<ColumnarValue> {
        make_scalar_function(greatest_inner)(args)
    }

    fn aliases(&self) -> &[String] {
        &self.aliases
    }

    fn documentation(&self) -> Option<&Documentation> {
        Some(get_greatest_doc())
    }
}

static DOCUMENTATION: OnceLock<Documentation> = OnceLock::new();

fn get_greatest_doc() -> &'static Documentation {
    DOCUMENTATION.get_or_init(|| {
        Documentation::builder()
            .with_description(
                "Returns the greatest value among the arguments, skipping null values.",
            )
            .with_syntax_example("greatest(value1[, value2[, ...]])")
            .with_sql_example(
                r#"
sql
> SELECT greatest(10, 20, 30);
+----------------------+
| greatest(10, 20, 30) |
+----------------------+
| 30                   |
+----------------------+
"#,
            )
            .build()
            .unwrap()
    })
}

pub fn greatest_inner(args: &[ArrayRef]) -> Result<ArrayRef> {
    if args.is_empty() {
        return Err(DataFusionError::Plan(
            "The 'greatest' function requires at least one argument".to_string(),
        ));
    }

    // Determine the common supertype of all arguments
    let arg_types: Vec<DataType> =
        args.iter().map(|arg| arg.data_type().clone()).collect();
    let data_type = {
        let mut common_type = arg_types[0].clone();
        for arg_type in &arg_types[1..] {
            common_type = get_wider_type(&common_type, arg_type)?;
        }
        common_type
    };

    // Cast all arrays to the common type
    let arrays = args
        .iter()
        .map(|array| {
            arrow::compute::cast(array, &data_type).map_err(DataFusionError::from)
        })
        .collect::<Result<Vec<_>>>()?;

    if arrays.is_empty() || arrays[0].len() == 0 {
        return Err(DataFusionError::Execution(
            "The input arrays are empty".to_string(),
        ));
    }

    // Implement the logic for different data types
    match data_type {
        DataType::Int8 => compute_greatest_numeric::<Int8Type>(&arrays),
        DataType::Int16 => compute_greatest_numeric::<Int16Type>(&arrays),
        DataType::Int32 => compute_greatest_numeric::<Int32Type>(&arrays),
        DataType::Int64 => compute_greatest_numeric::<Int64Type>(&arrays),
        DataType::UInt8 => compute_greatest_numeric::<UInt8Type>(&arrays),
        DataType::UInt16 => compute_greatest_numeric::<UInt16Type>(&arrays),
        DataType::UInt32 => compute_greatest_numeric::<UInt32Type>(&arrays),
        DataType::UInt64 => compute_greatest_numeric::<UInt64Type>(&arrays),
        DataType::Float32 => compute_greatest_numeric::<Float32Type>(&arrays),
        DataType::Float64 => compute_greatest_numeric::<Float64Type>(&arrays),
        DataType::Utf8 => compute_greatest_utf8(&arrays),
        DataType::LargeUtf8 => compute_greatest_large_utf8(&arrays),
        DataType::List(_) => compute_greatest_list(&arrays),
        DataType::Struct(_) => compute_greatest_struct(&arrays),
        _ => Err(DataFusionError::NotImplemented(format!(
            "Greatest function not implemented for data type {:?}",
            data_type
        ))),
    }
}

// Generic function to compute greatest for numeric types
fn compute_greatest_numeric<T>(arrays: &[ArrayRef]) -> Result<ArrayRef>
where
    T: ArrowPrimitiveType,
    T::Native: PartialOrd + Copy,
{
    let arrays = arrays
        .iter()
        .map(|array| {
            array
                .as_any()
                .downcast_ref::<PrimitiveArray<T>>()
                .ok_or_else(|| {
                    DataFusionError::Execution("Failed to downcast array".to_string())
                })
        })
        .collect::<Result<Vec<_>>>()?;

    let num_rows = arrays[0].len();
    let mut builder = PrimitiveBuilder::<T>::with_capacity(num_rows);

    for row in 0..num_rows {
        let mut max_value: Option<T::Native> = None;
        for array in &arrays {
            if array.is_valid(row) {
                let value = array.value(row);
                if value.partial_cmp(&value).is_none() {
                    // Skip NaN values
                    continue;
                }
                max_value = Some(match max_value {
                    None => value,
                    Some(current_max) => {
                        if value > current_max {
                            value
                        } else {
                            current_max
                        }
                    }
                });
            }
        }
        if let Some(value) = max_value {
            builder.append_value(value);
        } else {
            builder.append_null();
        }
    }

    Ok(Arc::new(builder.finish()))
}

fn compute_greatest_utf8(arrays: &[ArrayRef]) -> Result<ArrayRef> {
    let arrays = arrays
        .iter()
        .map(|array| {
            array.as_any().downcast_ref::<StringArray>().ok_or_else(|| {
                DataFusionError::Execution("Failed to downcast array".to_string())
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let num_rows = arrays[0].len();
    let mut builder = StringBuilder::new();

    for row in 0..num_rows {
        let mut max_value: Option<&str> = None;
        for array in &arrays {
            if array.is_valid(row) {
                let value = array.value(row);
                max_value = Some(match max_value {
                    None => value,
                    Some(current_max) => {
                        if value > current_max {
                            value
                        } else {
                            current_max
                        }
                    }
                });
            }
        }
        if let Some(value) = max_value {
            builder.append_value(value);
        } else {
            builder.append_null();
        }
    }

    Ok(Arc::new(builder.finish()))
}

fn compute_greatest_large_utf8(arrays: &[ArrayRef]) -> Result<ArrayRef> {
    let arrays = arrays
        .iter()
        .map(|array| {
            array
                .as_any()
                .downcast_ref::<LargeStringArray>()
                .ok_or_else(|| {
                    DataFusionError::Execution("Failed to downcast array".to_string())
                })
        })
        .collect::<Result<Vec<_>>>()?;

    let num_rows = arrays[0].len();
    let mut builder = LargeStringBuilder::new();

    for row in 0..num_rows {
        let mut max_value: Option<&str> = None;
        for array in &arrays {
            if array.is_valid(row) {
                let value = array.value(row);
                max_value = Some(match max_value {
                    None => value,
                    Some(current_max) => {
                        if value > current_max {
                            value
                        } else {
                            current_max
                        }
                    }
                });
            }
        }
        if let Some(value) = max_value {
            builder.append_value(value);
        } else {
            builder.append_null();
        }
    }

    Ok(Arc::new(builder.finish()))
}

fn compute_greatest_list(arrays: &[ArrayRef]) -> Result<ArrayRef> {
    if arrays.is_empty() {
        return Err(DataFusionError::Execution(
            "No arrays provided for greatest_list computation".to_string(),
        ));
    }

    // Determine the element type from the first array
    let first_array =
        arrays[0]
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| {
                DataFusionError::Execution("Failed to downcast to ListArray".to_string())
            })?;

    let element_type = first_array.value_type().clone();

    // Create a builder based on the element type
    let mut builder = ListBuilder::new(make_builder(&element_type, arrays[0].len())?);

    for row in 0..arrays[0].len() {
        let mut max_element: Option<ColumnarValue> = None;

        for array in arrays {
            if array.is_valid(row) {
                let list =
                    array.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
                        DataFusionError::Execution(
                            "Failed to downcast to ListArray".to_string(),
                        )
                    })?;

                let elements = list.value(row);
                let element_array = elements.clone(); // Clone to avoid borrowing issues

                // Determine the greatest element in the list
                let current_max = greatest_inner(&[element_array.clone()])?;
                max_element = match (max_element, current_max) {
                    (None, ColumnarValue::Array(arr)) => {
                        Some(ColumnarValue::Array(arr.clone()))
                    }
                    (
                        Some(ColumnarValue::Array(existing_arr)),
                        ColumnarValue::Array(new_arr),
                    ) => {
                        // Compare existing_arr and new_arr
                        // Assuming single-element arrays for simplicity
                        if existing_arr.value(0) < new_arr.value(0) {
                            Some(ColumnarValue::Array(new_arr.clone()))
                        } else {
                            Some(ColumnarValue::Array(existing_arr.clone()))
                        }
                    }
                    (Some(_), ColumnarValue::Scalar(_)) => max_element, // Ignore scalar for lists
                    _ => max_element,
                };
            }
        }

        if let Some(ColumnarValue::Array(arr)) = max_element {
            // Append the array to the builder
            builder.values().append_array(&arr)?;
            builder.append(true)?;
        } else {
            builder.append(false);
        }
    }

    Ok(Arc::new(builder.finish()))
}

fn compute_greatest_struct(arrays: &[ArrayRef]) -> Result<ArrayRef> {
    if arrays.is_empty() {
        return Err(DataFusionError::Execution(
            "No arrays provided for greatest_struct computation".to_string(),
        ));
    }

    // Ensure all arrays are StructArrays and have the same schema
    let first_array = arrays[0]
        .as_any()
        .downcast_ref::<StructArray>()
        .ok_or_else(|| {
            DataFusionError::Execution("Failed to downcast to StructArray".to_string())
        })?;

    let fields = first_array.fields();
    let num_fields = fields.len();
    let num_rows = first_array.len();

    // Create builders for each field based on their data types
    let mut field_builders: Vec<Box<dyn ArrayBuilder>> = fields
        .iter()
        .map(|field| make_builder(field.data_type(), num_rows))
        .collect::<Result<_>>()?;

    let mut struct_builder = StructBuilder::new(fields.clone(), field_builders);

    for row in 0..num_rows {
        let mut append_null = true;

        for field_idx in 0..num_fields {
            // Gather all arrays for this field across input arrays
            let field_arrays = arrays
                .iter()
                .map(|array| {
                    let struct_array = array
                        .as_any()
                        .downcast_ref::<StructArray>()
                        .ok_or_else(|| {
                            DataFusionError::Execution(
                                "Failed to downcast to StructArray".to_string(),
                            )
                        })?;
                    Ok(struct_array.column(field_idx).clone())
                })
                .collect::<Result<Vec<_>>>()?;

            // Apply greatest_inner on the field arrays
            let greatest = greatest_inner(&field_arrays)?;

            match greatest {
                ColumnarValue::Array(arr) => {
                    // Determine the field's data type
                    let field_type = &fields[field_idx].data_type();

                    // Downcast the array to its concrete type based on field_type
                    match field_type {
                        DataType::Int8 => {
                            let arr = arr
                                .as_any()
                                .downcast_ref::<Int8Array>()
                                .ok_or_else(|| {
                                    DataFusionError::Execution(
                                        "Failed to downcast to Int8Array".to_string(),
                                    )
                                })?;
                            struct_builder
                                .field_builder::<Int8Builder>(field_idx)?
                                .append_option(arr.get(row))?;
                        }
                        DataType::Int16 => {
                            let arr = arr
                                .as_any()
                                .downcast_ref::<Int16Array>()
                                .ok_or_else(|| {
                                    DataFusionError::Execution(
                                        "Failed to downcast to Int16Array".to_string(),
                                    )
                                })?;
                            struct_builder
                                .field_builder::<Int16Builder>(field_idx)?
                                .append_option(arr.get(row))?;
                        }
                        DataType::Int32 => {
                            let arr = arr
                                .as_any()
                                .downcast_ref::<Int32Array>()
                                .ok_or_else(|| {
                                    DataFusionError::Execution(
                                        "Failed to downcast to Int32Array".to_string(),
                                    )
                                })?;
                            struct_builder
                                .field_builder::<Int32Builder>(field_idx)?
                                .append_option(arr.get(row))?;
                        }
                        DataType::Int64 => {
                            let arr = arr
                                .as_any()
                                .downcast_ref::<Int64Array>()
                                .ok_or_else(|| {
                                    DataFusionError::Execution(
                                        "Failed to downcast to Int64Array".to_string(),
                                    )
                                })?;
                            struct_builder
                                .field_builder::<Int64Builder>(field_idx)?
                                .append_option(arr.get(row))?;
                        }
                        DataType::UInt8 => {
                            let arr = arr
                                .as_any()
                                .downcast_ref::<UInt8Array>()
                                .ok_or_else(|| {
                                    DataFusionError::Execution(
                                        "Failed to downcast to UInt8Array".to_string(),
                                    )
                                })?;
                            struct_builder
                                .field_builder::<UInt8Builder>(field_idx)?
                                .append_option(arr.get(row))?;
                        }
                        DataType::UInt16 => {
                            let arr = arr
                                .as_any()
                                .downcast_ref::<UInt16Array>()
                                .ok_or_else(|| {
                                    DataFusionError::Execution(
                                        "Failed to downcast to UInt16Array".to_string(),
                                    )
                                })?;
                            struct_builder
                                .field_builder::<UInt16Builder>(field_idx)?
                                .append_option(arr.get(row))?;
                        }
                        DataType::UInt32 => {
                            let arr = arr
                                .as_any()
                                .downcast_ref::<UInt32Array>()
                                .ok_or_else(|| {
                                    DataFusionError::Execution(
                                        "Failed to downcast to UInt32Array".to_string(),
                                    )
                                })?;
                            struct_builder
                                .field_builder::<UInt32Builder>(field_idx)?
                                .append_option(arr.get(row))?;
                        }
                        DataType::UInt64 => {
                            let arr = arr
                                .as_any()
                                .downcast_ref::<UInt64Array>()
                                .ok_or_else(|| {
                                    DataFusionError::Execution(
                                        "Failed to downcast to UInt64Array".to_string(),
                                    )
                                })?;
                            struct_builder
                                .field_builder::<UInt64Builder>(field_idx)?
                                .append_option(arr.get(row))?;
                        }
                        DataType::Float32 => {
                            let arr = arr
                                .as_any()
                                .downcast_ref::<Float32Array>()
                                .ok_or_else(|| {
                                    DataFusionError::Execution(
                                        "Failed to downcast to Float32Array".to_string(),
                                    )
                                })?;
                            struct_builder
                                .field_builder::<Float32Builder>(field_idx)?
                                .append_option(arr.get(row))?;
                        }
                        DataType::Float64 => {
                            let arr = arr
                                .as_any()
                                .downcast_ref::<Float64Array>()
                                .ok_or_else(|| {
                                    DataFusionError::Execution(
                                        "Failed to downcast to Float64Array".to_string(),
                                    )
                                })?;
                            struct_builder
                                .field_builder::<Float64Builder>(field_idx)?
                                .append_option(arr.get(row))?;
                        }
                        DataType::Utf8 => {
                            let arr = arr
                                .as_any()
                                .downcast_ref::<StringArray>()
                                .ok_or_else(|| {
                                    DataFusionError::Execution(
                                        "Failed to downcast to StringArray".to_string(),
                                    )
                                })?;
                            struct_builder
                                .field_builder::<StringBuilder>(field_idx)?
                                .append_option(arr.get(row))?;
                        }
                        DataType::LargeUtf8 => {
                            let arr = arr
                                .as_any()
                                .downcast_ref::<LargeStringArray>()
                                .ok_or_else(|| {
                                    DataFusionError::Execution(
                                        "Failed to downcast to LargeStringArray"
                                            .to_string(),
                                    )
                                })?;
                            struct_builder
                                .field_builder::<LargeStringBuilder>(field_idx)?
                                .append_option(arr.get(row))?;
                        }
                        // Add more data types as needed
                        _ => {
                            return Err(DataFusionError::NotImplemented(format!(
                                "Greatest function not implemented for struct field data type {:?}",
                                field_type
                            )));
                        }
                    }
                    append_null = false;
                }
                ColumnarValue::Scalar(_) => {
                    // Handle scalar if needed
                    // For simplicity, we'll skip scalar handling in structs
                }
            }
        }

        // After processing all fields for the row
        struct_builder.append(append_null)?;
    }

    Ok(Arc::new(struct_builder.finish()))
}

fn make_builder(data_type: &DataType, capacity: usize) -> Result<Box<dyn ArrayBuilder>> {
    Ok(match data_type {
        DataType::Int8 => Box::new(Int8Builder::with_capacity(capacity)),
        DataType::Int16 => Box::new(Int16Builder::with_capacity(capacity)),
        DataType::Int32 => Box::new(Int32Builder::with_capacity(capacity)),
        DataType::Int64 => Box::new(Int64Builder::with_capacity(capacity)),
        DataType::UInt8 => Box::new(UInt8Builder::with_capacity(capacity)),
        DataType::UInt16 => Box::new(UInt16Builder::with_capacity(capacity)),
        DataType::UInt32 => Box::new(UInt32Builder::with_capacity(capacity)),
        DataType::UInt64 => Box::new(UInt64Builder::with_capacity(capacity)),
        DataType::Float32 => Box::new(Float32Builder::with_capacity(capacity)),
        DataType::Float64 => Box::new(Float64Builder::with_capacity(capacity)),
        DataType::Utf8 => Box::new(StringBuilder::with_capacity(capacity, 0)),
        DataType::LargeUtf8 => Box::new(LargeStringBuilder::with_capacity(capacity, 0)),
        _ => {
            return Err(DataFusionError::NotImplemented(format!(
                "Builder not implemented for data type {:?}",
                data_type
            )))
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::*;
    use datafusion_common::Result;
    use std::sync::Arc;

    #[test]
    fn test_greatest_int32() -> Result<()> {
        let input_int = vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, None])) as ArrayRef,
            Arc::new(Int32Array::from(vec![4, None, 6, 8])) as ArrayRef,
            Arc::new(Int32Array::from(vec![7, 5, None, 9])) as ArrayRef,
        ];

        let greatest = greatest_inner(&input_int)?;
        let greatest_int = greatest
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Failed to downcast to Int32Array");

        let expected = Int32Array::from(vec![7, 5, 6, 9]);
        assert_eq!(greatest_int, &expected);

        Ok(())
    }

    #[test]
    fn test_greatest_float64_with_nan() -> Result<()> {
        let input_float = vec![
            Arc::new(Float64Array::from(vec![1.1, f64::NAN, 3.3])) as ArrayRef,
            Arc::new(Float64Array::from(vec![4.4, 5.5, f64::NAN])) as ArrayRef,
            Arc::new(Float64Array::from(vec![7.7, 8.8, 9.9])) as ArrayRef,
        ];

        let greatest = greatest_inner(&input_float)?;
        let greatest_float = greatest
            .as_any()
            .downcast_ref::<Float64Array>()
            .expect("Failed to downcast to Float64Array");

        let expected = Float64Array::from(vec![7.7, 8.8, 9.9]);
        assert_eq!(greatest_float, &expected);

        Ok(())
    }

    #[test]
    fn test_greatest_utf8() -> Result<()> {
        let input_utf8 = vec![
            Arc::new(StringArray::from(vec!["apple", "banana", "cherry"])) as ArrayRef,
            Arc::new(StringArray::from(vec!["apricot", "blueberry", "citrus"]))
                as ArrayRef,
            Arc::new(StringArray::from(vec![
                "avocado",
                "blackberry",
                "cranberry",
            ])) as ArrayRef,
        ];

        let greatest = greatest_inner(&input_utf8)?;
        let greatest_utf8 = greatest
            .as_any()
            .downcast_ref::<StringArray>()
            .expect("Failed to downcast to StringArray");

        let expected = StringArray::from(vec!["avocado", "blackberry", "citrus"]);
        assert_eq!(greatest_utf8, &expected);

        Ok(())
    }

    #[test]
    fn test_greatest_with_nulls() -> Result<()> {
        let input_int = vec![
            Arc::new(Int32Array::from(vec![None, None, None])) as ArrayRef,
            Arc::new(Int32Array::from(vec![None, 2, None])) as ArrayRef,
            Arc::new(Int32Array::from(vec![None, None, 3])) as ArrayRef,
        ];

        let greatest = greatest_inner(&input_int)?;
        let greatest_int = greatest
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Failed to downcast to Int32Array");

        let expected = Int32Array::from(vec![None, 2, 3]);
        assert_eq!(greatest_int, &expected);

        Ok(())
    }

    #[test]
    fn test_greatest_mixed_types_error() -> Result<()> {
        let input_mixed = vec![
            Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
            Arc::new(StringArray::from(vec!["a", "b", "c"])) as ArrayRef,
        ];

        let result = greatest_inner(&input_mixed);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "Error during type coercion: Failed to get wider type between Int32 and Utf8"
        );

        Ok(())
    }

    #[test]
    fn test_greatest_empty_input() {
        let input_empty: Vec<ArrayRef> = vec![];

        let result = greatest_inner(&input_empty);
        assert!(result.is_err());
        assert_eq!(
            result.unwrap_err().to_string(),
            "The 'greatest' function requires at least one argument"
        );
    }

    #[test]
    fn test_greatest_single_argument() -> Result<()> {
        let input_single = vec![Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef];

        let greatest = greatest_inner(&input_single)?;
        let greatest_int = greatest
            .as_any()
            .downcast_ref::<Int32Array>()
            .expect("Failed to downcast to Int32Array");

        let expected = Int32Array::from(vec![1, 2, 3]);
        assert_eq!(greatest_int, &expected);

        Ok(())
    }

    #[test]
    fn test_greatest_large_utf8() -> Result<()> {
        let input_large_utf8 = vec![
            Arc::new(LargeStringArray::from(vec!["alpha", "beta", "gamma"])) as ArrayRef,
            Arc::new(LargeStringArray::from(vec!["delta", "epsilon", "zeta"]))
                as ArrayRef,
            Arc::new(LargeStringArray::from(vec!["eta", "theta", "iota"])) as ArrayRef,
        ];

        let greatest = greatest_inner(&input_large_utf8)?;
        let greatest_large_utf8 = greatest
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .expect("Failed to downcast to LargeStringArray");

        let expected = LargeStringArray::from(vec!["eta", "theta", "zeta"]);
        assert_eq!(greatest_large_utf8, &expected);

        Ok(())
    }

    #[test]
    fn test_greatest_list_int32() -> Result<()> {
        let list1 = Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(1), Some(2), Some(3)]),
            Some(vec![Some(4), Some(5), Some(6)]),
            None,
        ])) as ArrayRef;

        let list2 = Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(7), Some(8), Some(9)]),
            Some(vec![Some(10), Some(11), Some(12)]),
            Some(vec![Some(13), Some(14)]),
        ])) as ArrayRef;

        let input_list = vec![list1, list2];
        let greatest_list = greatest_inner(&input_list)?;

        // Since ListArray's display is not straightforward, we'll perform type-specific assertions
        let expected = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(7), Some(8), Some(9)]),
            Some(vec![Some(10), Some(11), Some(12)]),
            Some(vec![Some(13), Some(14)]),
        ]);

        assert_eq!(greatest_list, expected.as_ref());

        Ok(())
    }

    #[test]
    fn test_greatest_struct_int64() -> Result<()> {
        // Define a StructArray with two fields: a and b
        let field_a = Arc::new(Int64Array::from(vec![1, 2, 3])) as ArrayRef;
        let field_b = Arc::new(Int64Array::from(vec![4, 5, 6])) as ArrayRef;
        let struct_array1 = Arc::new(StructArray::from(vec![
            ("a", field_a.clone()),
            ("b", field_b.clone()),
        ])) as ArrayRef;

        let field_a2 = Arc::new(Int64Array::from(vec![7, 8, 9])) as ArrayRef;
        let field_b2 = Arc::new(Int64Array::from(vec![10, 11, 12])) as ArrayRef;
        let struct_array2 = Arc::new(StructArray::from(vec![
            ("a", field_a2.clone()),
            ("b", field_b2.clone()),
        ])) as ArrayRef;

        let input_struct = vec![struct_array1, struct_array2];
        let greatest_struct = greatest_inner(&input_struct)?;

        // Expected StructArray with greatest values
        let expected_field_a = Int64Array::from(vec![7, 8, 9]);
        let expected_field_b = Int64Array::from(vec![10, 11, 12]);
        let expected_struct = StructArray::from(vec![
            ("a", Arc::new(expected_field_a) as ArrayRef),
            ("b", Arc::new(expected_field_b) as ArrayRef),
        ]);

        assert_eq!(greatest_struct, expected_struct.as_ref());

        Ok(())
    }
}
