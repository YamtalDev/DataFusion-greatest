// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! [`ScalarUDFImpl`] definitions for the `greatest` function.

use crate::utils::make_scalar_function;
use arrow::array::*;
use arrow::datatypes::DataType;
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
    "Returns the greatest value among the arguments.",
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
            // Remove or comment out this line
            // .with_doc_section(DOC_SECTION_GENERAL)
            .with_description("Returns the greatest value among the arguments.")
            .with_syntax_example("greatest(value1[, value2[, ...]])")
            .with_sql_example(
                r#"```sql
> SELECT greatest(10, 20, 30);
+----------------------+
| greatest(10, 20, 30) |
+----------------------+
| 30                   |
+----------------------+
```"#,
            )
            .build()
            .unwrap()
    })
}

fn greatest_inner(args: &[ArrayRef]) -> Result<ArrayRef> {
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

    // Implement the logic for different data types
    match data_type {
        DataType::Int32 => compute_greatest_int32(&arrays),
        DataType::Float64 => compute_greatest_float64(&arrays),
        DataType::Utf8 => compute_greatest_utf8(&arrays),
        _ => Err(DataFusionError::NotImplemented(format!(
            "Greatest function not implemented for data type {:?}",
            data_type
        ))),
    }
}

fn compute_greatest_int32(arrays: &[ArrayRef]) -> Result<ArrayRef> {
    let arrays = arrays
        .iter()
        .map(|array| array.as_any().downcast_ref::<Int32Array>().unwrap())
        .collect::<Vec<_>>();

    let num_rows = arrays[0].len();
    let mut builder = Int32Builder::with_capacity(num_rows);

    for row in 0..num_rows {
        let mut max_value: Option<i32> = None;
        for array in &arrays {
            if array.is_valid(row) {
                let value = array.value(row);
                max_value =
                    Some(max_value.map_or(value, |current_max| current_max.max(value)));
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

fn compute_greatest_float64(arrays: &[ArrayRef]) -> Result<ArrayRef> {
    let arrays = arrays
        .iter()
        .map(|array| array.as_any().downcast_ref::<Float64Array>().unwrap())
        .collect::<Vec<_>>();

    let num_rows = arrays[0].len();
    let mut builder = Float64Builder::with_capacity(num_rows);

    for row in 0..num_rows {
        let mut max_value: Option<f64> = None;
        for array in &arrays {
            if array.is_valid(row) {
                let value = array.value(row);
                max_value =
                    Some(max_value.map_or(value, |current_max| current_max.max(value)));
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
        .map(|array| array.as_any().downcast_ref::<StringArray>().unwrap())
        .collect::<Vec<_>>();

    let num_rows = arrays[0].len();
    let mut builder = StringBuilder::with_capacity(num_rows, 0);

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
