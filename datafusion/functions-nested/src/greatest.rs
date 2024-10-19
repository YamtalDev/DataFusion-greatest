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
    // Assuming the list elements are Int32Array for simplicity
    // You may need to generalize this based on element type
    let arrays = arrays
        .iter()
        .map(|array| {
            array.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
                DataFusionError::Execution("Failed to downcast array".to_string())
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let num_rows = arrays[0].len();
    let value_builder = Int32Builder::new();
    let mut builder = ListBuilder::new(value_builder);

    for row in 0..num_rows {
        let mut max_value: Option<i32> = None;
        for array in &arrays {
            if array.is_valid(row) {
                let values = array.value(row);
                let value_array = values
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .ok_or_else(|| {
                        DataFusionError::Execution(
                            "Failed to downcast value array".to_string(),
                        )
                    })?;

                for index in 0..value_array.len() {
                    if value_array.is_valid(index) {
                        let value = value_array.value(index);
                        max_value = Some(
                            max_value.map_or(value, |current_max| current_max.max(value)),
                        );
                    }
                }
            }
        }

        if let Some(value) = max_value {
            builder.values().append_value(value);
            builder.append(true);
        } else {
            builder.append(false);
        }
    }

    Ok(Arc::new(builder.finish()))
}

fn compute_greatest_struct(arrays: &[ArrayRef]) -> Result<ArrayRef> {
    let arrays = arrays
        .iter()
        .map(|array| {
            array.as_any().downcast_ref::<StructArray>().ok_or_else(|| {
                DataFusionError::Execution("Failed to downcast array".to_string())
            })
        })
        .collect::<Result<Vec<_>>>()?;

    let num_rows = arrays[0].len();
    let fields = arrays[0].data_type().clone();
    let field_types = match fields {
        DataType::Struct(fields) => fields,
        _ => {
            return Err(DataFusionError::Execution(
                "Expected Struct data type".to_string(),
            ))
        }
    };

    // Create field builders
    let field_builders: Vec<Box<dyn ArrayBuilder>> = field_types
        .iter()
        .map(|field| make_builder(field.data_type(), num_rows))
        .collect::<Result<_>>()?;

    let mut builder = StructBuilder::new(field_types, field_builders);

    for row in 0..num_rows {
        let mut append_null = true;

        for field_index in 0..builder.num_fields() {
            let field_builder =
                builder.field_builder::<Int64Builder>(field_index).unwrap();
            let mut max_value: Option<i64> = None;

            for array in &arrays {
                let column = array.column(field_index);
                if column.is_valid(row) {
                    let value = column
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .ok_or_else(|| {
                            DataFusionError::Execution(
                                "Failed to downcast column array".to_string(),
                            )
                        })?
                        .value(row);

                    max_value = Some(
                        max_value.map_or(value, |current_max| current_max.max(value)),
                    );
                }
            }

            if let Some(value) = max_value {
                field_builder.append_value(value);
                append_null = false;
            } else {
                field_builder.append_null();
            }
        }

        if append_null {
            builder.append(false);
        } else {
            builder.append(true);
        }
    }

    Ok(Arc::new(builder.finish()))
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

// // Licensed to the Apache Software Foundation (ASF) under one
// // or more contributor license agreements.  See the NOTICE file
// // distributed with this work for additional information
// // regarding copyright ownership.  The ASF licenses this file
// // to you under the Apache License, Version 2.0 (the
// // "License"); you may not use this file except in compliance
// // with the License.  You may obtain a copy of the License at
// //
// //   http://www.apache.org/licenses/LICENSE-2.0
// //
// // Unless required by applicable law or agreed to in writing,
// // software distributed under the License is distributed on an
// // "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// // KIND, either express or implied.  See the License for the
// // specific language governing permissions and limitations
// // under the License.

// //! [`ScalarUDFImpl`] definitions for the `greatest` function.

// use crate::utils::make_scalar_function;
// use arrow::array::*;
// use arrow::datatypes::DataType;
// use datafusion_common::{DataFusionError, Result};
// use datafusion_expr::{
//     type_coercion::binary::get_wider_type, ColumnarValue, Documentation, ScalarUDFImpl,
//     Signature, Volatility,
// };
// use std::any::Any;
// use std::sync::{Arc, OnceLock};

// make_udf_expr_and_func!(
//     Greatest,
//     greatest,
//     "Returns the greatest value among the arguments, skipping nulls.",
//     greatest_udf
// );

// #[derive(Debug)]
// pub struct Greatest {
//     signature: Signature,
//     aliases: Vec<String>,
// }

// impl Greatest {
//     pub fn new() -> Self {
//         Self {
//             signature: Signature::variadic_any(Volatility::Immutable),
//             aliases: vec![],
//         }
//     }
// }

// impl ScalarUDFImpl for Greatest {
//     fn as_any(&self) -> &dyn Any {
//         self
//     }

//     fn name(&self) -> &str {
//         "greatest"
//     }

//     fn signature(&self) -> &Signature {
//         &self.signature
//     }

//     fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
//         if arg_types.is_empty() {
//             return Err(DataFusionError::Plan(
//                 "The 'greatest' function requires at least one argument".to_string(),
//             ));
//         }

//         // Find the common supertype among the arguments
//         let mut common_type = arg_types[0].clone();
//         for arg_type in &arg_types[1..] {
//             common_type = get_wider_type(&common_type, arg_type)?;
//         }

//         Ok(common_type)
//     }

//     fn invoke(&self, args: &[ColumnarValue]) -> Result<ColumnarValue> {
//         make_scalar_function(greatest_inner)(args)
//     }

//     fn aliases(&self) -> &[String] {
//         &self.aliases
//     }

//     fn documentation(&self) -> Option<&Documentation> {
//         Some(get_greatest_doc())
//     }
// }

// static DOCUMENTATION: OnceLock<Documentation> = OnceLock::new();

// fn get_greatest_doc() -> &'static Documentation {
//     DOCUMENTATION.get_or_init(|| {
//         Documentation::builder()
//             .with_description(
//                 "Returns the greatest value among the arguments, skipping null values.",
//             )
//             .with_syntax_example("greatest(value1[, value2[, ...]])")
//             .with_sql_example(
//                 r#"```sql
// > SELECT greatest(10, 20, 30);
// +----------------------+
// | greatest(10, 20, 30) |
// +----------------------+
// | 30                   |
// +----------------------+
// ```"#,
//             )
//             .build()
//             .unwrap()
//     })
// }

// fn greatest_inner(args: &[ArrayRef]) -> Result<ArrayRef> {
//     if args.is_empty() {
//         return Err(DataFusionError::Plan(
//             "The 'greatest' function requires at least one argument".to_string(),
//         ));
//     }

//     // Determine the common supertype of all arguments
//     let arg_types: Vec<DataType> =
//         args.iter().map(|arg| arg.data_type().clone()).collect();
//     let data_type = {
//         let mut common_type = arg_types[0].clone();
//         for arg_type in &arg_types[1..] {
//             common_type = get_wider_type(&common_type, arg_type)?;
//         }
//         common_type
//     };

//     // Cast all arrays to the common type
//     let arrays = args
//         .iter()
//         .map(|array| {
//             arrow::compute::cast(array, &data_type).map_err(DataFusionError::from)
//         })
//         .collect::<Result<Vec<_>>>()?;

//     // Implement the logic for different data types
//     match data_type {
//         DataType::Int32 => compute_greatest_int32(&arrays),
//         DataType::Int64 => compute_greatest_int64(&arrays),
//         DataType::Float64 => compute_greatest_float64(&arrays),
//         DataType::Utf8 => compute_greatest_utf8(&arrays),
//         DataType::LargeUtf8 => compute_greatest_large_utf8(&arrays),
//         _ => Err(DataFusionError::NotImplemented(format!(
//             "Greatest function not implemented for data type {:?}",
//             data_type
//         ))),
//     }
// }

// fn compute_greatest_int32(arrays: &[ArrayRef]) -> Result<ArrayRef> {
//     let arrays = arrays
//         .iter()
//         .map(|array| array.as_any().downcast_ref::<Int32Array>().unwrap())
//         .collect::<Vec<_>>();

//     let num_rows = arrays[0].len();
//     let mut builder = Int32Builder::with_capacity(num_rows);

//     for row in 0..num_rows {
//         let mut max_value: Option<i32> = None;
//         for array in &arrays {
//             if array.is_valid(row) {
//                 let value = array.value(row);
//                 max_value =
//                     Some(max_value.map_or(value, |current_max| current_max.max(value)));
//             }
//         }
//         if let Some(value) = max_value {
//             builder.append_value(value);
//         } else {
//             builder.append_null();
//         }
//     }

//     Ok(Arc::new(builder.finish()))
// }

// fn compute_greatest_int64(arrays: &[ArrayRef]) -> Result<ArrayRef> {
//     let arrays = arrays
//         .iter()
//         .map(|array| array.as_any().downcast_ref::<Int64Array>().unwrap())
//         .collect::<Vec<_>>();

//     let num_rows = arrays[0].len();
//     let mut builder = Int64Builder::with_capacity(num_rows);

//     for row in 0..num_rows {
//         let mut max_value: Option<i64> = None;
//         for array in &arrays {
//             if array.is_valid(row) {
//                 let value = array.value(row);
//                 max_value =
//                     Some(max_value.map_or(value, |current_max| current_max.max(value)));
//             }
//         }
//         if let Some(value) = max_value {
//             builder.append_value(value);
//         } else {
//             builder.append_null();
//         }
//     }

//     Ok(Arc::new(builder.finish()))
// }

// fn compute_greatest_float64(arrays: &[ArrayRef]) -> Result<ArrayRef> {
//     let arrays = arrays
//         .iter()
//         .map(|array| array.as_any().downcast_ref::<Float64Array>().unwrap())
//         .collect::<Vec<_>>();

//     let num_rows = arrays[0].len();
//     let mut builder = Float64Builder::with_capacity(num_rows);

//     for row in 0..num_rows {
//         let mut max_value: Option<f64> = None;
//         for array in &arrays {
//             if array.is_valid(row) {
//                 let value = array.value(row);
//                 max_value =
//                     Some(max_value.map_or(value, |current_max| current_max.max(value)));
//             }
//         }
//         if let Some(value) = max_value {
//             builder.append_value(value);
//         } else {
//             builder.append_null();
//         }
//     }

//     Ok(Arc::new(builder.finish()))
// }

// fn compute_greatest_utf8(arrays: &[ArrayRef]) -> Result<ArrayRef> {
//     let arrays = arrays
//         .iter()
//         .map(|array| array.as_any().downcast_ref::<StringArray>().unwrap())
//         .collect::<Vec<_>>();

//     let num_rows = arrays[0].len();
//     let mut builder = StringBuilder::with_capacity(num_rows, 0);

//     for row in 0..num_rows {
//         let mut max_value: Option<&str> = None;
//         for array in &arrays {
//             if array.is_valid(row) {
//                 let value = array.value(row);
//                 max_value = Some(match max_value {
//                     None => value,
//                     Some(current_max) => {
//                         if value > current_max {
//                             value
//                         } else {
//                             current_max
//                         }
//                     }
//                 });
//             }
//         }
//         if let Some(value) = max_value {
//             builder.append_value(value);
//         } else {
//             builder.append_null();
//         }
//     }

//     Ok(Arc::new(builder.finish()))
// }

// fn compute_greatest_large_utf8(arrays: &[ArrayRef]) -> Result<ArrayRef> {
//     let arrays = arrays
//         .iter()
//         .map(|array| array.as_any().downcast_ref::<LargeStringArray>().unwrap())
//         .collect::<Vec<_>>();

//     let num_rows = arrays[0].len();
//     let mut builder = LargeStringBuilder::with_capacity(num_rows, 0);

//     for row in 0..num_rows {
//         let mut max_value: Option<&str> = None;
//         for array in &arrays {
//             if array.is_valid(row) {
//                 let value = array.value(row);
//                 max_value = Some(match max_value {
//                     None => value,
//                     Some(current_max) => {
//                         if value > current_max {
//                             value
//                         } else {
//                             current_max
//                         }
//                     }
//                 });
//             }
//         }
//         if let Some(value) = max_value {
//             builder.append_value(value);
//         } else {
//             builder.append_null();
//         }
//     }

//     Ok(Arc::new(builder.finish()))
// }
