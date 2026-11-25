use std::{
    cell::{Cell, OnceCell, UnsafeCell},
    collections::HashMap,
    ffi::{CStr, CString},
    os::raw::c_char,
    ptr::NonNull,
    sync::Arc,
};

use arrow::{
    array::StructArray,
    datatypes::{Schema, SchemaRef},
    ffi::{FFI_ArrowArray, FFI_ArrowSchema, from_ffi},
};

use super::{Result, ffi};
use crate::{
    Error,
    core::{LogicalTypeHandle, LogicalTypeId},
    error::{
        arrow_conversion_failure, duckdb_failure_from_message, result_error_message, result_from_duckdb_error_data,
    },
    types::Type,
};
#[cfg(feature = "polars")]
use polars_core::utils::arrow as polars_arrow;

#[derive(Debug)]
pub(crate) struct ExecutedResult {
    arrow_options: ArrowOptionsHandle,
    result: DuckdbResultHandle,
    schema: SchemaRef,
    columns: Box<[ResultColumn]>,
    arrow_schema: FFI_ArrowSchema,
    exhausted: Cell<bool>,
    column_name_cache: OnceCell<HashMap<Box<str>, usize>>,
    #[cfg(feature = "polars")]
    polars_arrow_field: OnceCell<polars_arrow::datatypes::Field>,
    #[cfg(feature = "polars")]
    polars_narrowed_type: OnceCell<Option<arrow::datatypes::DataType>>,
}

// SAFETY: `ExecutedResult` is Send but not Sync. It owns its DuckDB result in an
// UnsafeCell because fetching mutates DuckDB's result cursor through `&self`.
// Cell and OnceCell keep cache/cursor mutation local to the statement. Moving
// the owner to another thread transfers the only access path to the result and
// Arrow options handles; concurrent access is not promised.
unsafe impl Send for ExecutedResult {}

#[derive(Debug)]
struct ResultColumn {
    name: CString,
    logical_type: LogicalTypeHandle,
}

impl ExecutedResult {
    /// Takes ownership of a DuckDB result.
    ///
    /// # Safety
    ///
    /// `result` must be initialized by DuckDB and not owned anywhere else.
    /// This function takes ownership unconditionally: after it is called, the
    /// result is destroyed by `ExecutedResult` even if metadata loading fails.
    pub(crate) unsafe fn new(result: ffi::duckdb_result) -> Result<Self> {
        let result = DuckdbResultHandle::new(result);
        let arrow_options = unsafe { ArrowOptionsHandle::new(&result)? };
        let (schema, columns, arrow_schema) = unsafe { Self::load_schema_and_columns(&result, &arrow_options)? };
        Ok(Self {
            arrow_options,
            result,
            schema,
            columns,
            arrow_schema,
            exhausted: Cell::new(false),
            column_name_cache: OnceCell::new(),
            #[cfg(feature = "polars")]
            polars_arrow_field: OnceCell::new(),
            #[cfg(feature = "polars")]
            polars_narrowed_type: OnceCell::new(),
        })
    }

    pub(crate) fn row_count(&self) -> usize {
        unsafe { ffi::duckdb_row_count(self.result.as_mut_ptr()) as usize }
    }

    pub(crate) fn step(&self) -> Result<Option<StructArray>> {
        unsafe {
            let Some(chunk) = self.next_chunk()? else {
                return Ok(None);
            };
            self.data_chunk_to_struct_array(chunk).map(Some)
        }
    }

    #[cfg(feature = "polars")]
    pub(crate) fn step_polars(&self) -> Result<Option<polars_arrow::array::StructArray>> {
        unsafe {
            let Some(chunk) = self.next_chunk()? else {
                return Ok(None);
            };

            let mut ffi_arrow_array = polars_arrow::ffi::ArrowArray::empty();
            let array = if let Some(narrowed) = self.polars_narrowed_type()? {
                // Polars has no 256-bit decimal. Narrow those columns to the
                // 128-bit decimal exactly; a value that does not fit is an
                // error.
                let wide = self.data_chunk_to_struct_array(chunk)?;
                let options = arrow::compute::CastOptions {
                    safe: false,
                    ..Default::default()
                };
                let narrow = arrow::compute::cast_with_options(&wide, narrowed, &options).map_err(|err| {
                    arrow_conversion_failure("Could not narrow a DECIMAL of more than 38 digits for Polars", err)
                })?;
                let (array, schema) = arrow::ffi::to_ffi(&narrow.to_data())
                    .map_err(|err| arrow_conversion_failure("Could not export Arrow array to Polars", err))?;
                // Both structs are the Arrow C data interface `ArrowArray`;
                // the empty one owns nothing, so it is overwritten undropped.
                std::ptr::write(&mut ffi_arrow_array as *mut _ as *mut FFI_ArrowArray, array);
                let field = polars_arrow::ffi::import_field_from_c(
                    &*(&schema as *const FFI_ArrowSchema as *const polars_arrow::ffi::ArrowSchema),
                )
                .map_err(|err| arrow_conversion_failure("Could not import Polars Arrow field", err))?;
                polars_arrow::ffi::import_array_from_c(ffi_arrow_array, field.dtype)
            } else {
                self.chunk_to_arrow(chunk, &mut ffi_arrow_array as *mut _ as *mut ffi::ArrowArray)?;
                let field = self.polars_arrow_field()?;
                polars_arrow::ffi::import_array_from_c(ffi_arrow_array, field.dtype.clone())
            }
            .map_err(|err| arrow_conversion_failure("Could not import Polars Arrow array from C", err))?;
            let struct_array = array
                .as_any()
                .downcast_ref::<polars_arrow::array::StructArray>()
                .ok_or_else(|| duckdb_failure_from_message("Failed to downcast Polars Arrow array to StructArray"))?
                .to_owned();

            Ok(Some(struct_array))
        }
    }

    pub(crate) fn schema_ref(&self) -> &SchemaRef {
        &self.schema
    }

    pub(crate) fn result_column_logical_id(&self, idx: usize) -> LogicalTypeId {
        self.columns
            .get(idx)
            .unwrap_or_else(|| panic!("result column logical-type index {idx} is out of bounds"))
            .logical_type
            .id()
    }

    pub(crate) fn try_column_logical_type(&self, idx: usize) -> Result<LogicalTypeHandle> {
        unsafe { Self::try_result_column_logical_type(self.result.as_mut_ptr(), idx) }
    }

    pub(crate) fn column_name(&self, idx: usize) -> Option<&String> {
        self.schema.fields().get(idx).map(|field| field.name())
    }

    pub(crate) fn column_index(&self, name: &str) -> Option<usize> {
        let cache = self.column_name_cache.get_or_init(|| self.build_column_name_cache());
        cache.get(&*name.to_ascii_lowercase()).copied()
    }

    fn build_column_name_cache(&self) -> HashMap<Box<str>, usize> {
        let schema = self.schema_ref();
        let mut cache = HashMap::with_capacity(schema.fields().len());
        for (index, field) in schema.fields().iter().enumerate() {
            cache
                .entry(field.name().to_ascii_lowercase().into_boxed_str())
                .or_insert(index);
        }
        cache
    }

    unsafe fn load_schema_and_columns(
        result: &DuckdbResultHandle,
        arrow_options: &ArrowOptionsHandle,
    ) -> Result<(SchemaRef, Box<[ResultColumn]>, FFI_ArrowSchema)> {
        unsafe {
            let mut arrow_schema = FFI_ArrowSchema::empty();
            let columns = Self::load_result_columns(result.as_mut_ptr())?;
            Self::export_arrow_schema(
                arrow_options,
                &columns,
                &mut arrow_schema as *mut _ as *mut ffi::ArrowSchema,
            )?;
            let schema = Arc::new(
                Schema::try_from(&arrow_schema)
                    .map_err(|err| arrow_conversion_failure("Could not import DuckDB Arrow schema", err))?,
            );
            debug_assert_eq!(
                schema.fields().len(),
                columns.len(),
                "result schema and column cache are built from the same DuckDB column count"
            );

            Ok((schema, columns, arrow_schema))
        }
    }

    unsafe fn load_result_columns(result: *mut ffi::duckdb_result) -> Result<Box<[ResultColumn]>> {
        unsafe {
            let column_count = ffi::duckdb_column_count(result) as usize;
            let mut columns = Vec::with_capacity(column_count);

            for idx in 0..column_count {
                let logical_type = Self::try_result_column_logical_type(result, idx)?;
                reject_unsupported_result_logical_type(idx, &logical_type)?;

                let name = ffi::duckdb_column_name(result, idx as u64);
                if name.is_null() {
                    return Err(duckdb_failure_from_message(format!(
                        "Could not retrieve name for result column at index {idx}"
                    )));
                }
                let name = CStr::from_ptr(name).to_owned();

                columns.push(ResultColumn { name, logical_type });
            }

            Ok(columns.into_boxed_slice())
        }
    }

    unsafe fn try_result_column_logical_type(result: *mut ffi::duckdb_result, idx: usize) -> Result<LogicalTypeHandle> {
        unsafe {
            let ptr = ffi::duckdb_column_logical_type(result, idx as u64);
            logical_type_from_duckdb_column(ptr, idx)
        }
    }

    unsafe fn export_arrow_schema(
        arrow_options: &ArrowOptionsHandle,
        columns: &[ResultColumn],
        dst: *mut ffi::ArrowSchema,
    ) -> Result<()> {
        unsafe {
            let mut type_ptrs = columns.iter().map(|column| column.logical_type.ptr).collect::<Vec<_>>();
            let mut names = columns
                .iter()
                .map(|column| column.name.as_ptr() as *const c_char)
                .collect::<Vec<_>>();
            let error_data = ffi::duckdb_to_arrow_schema(
                arrow_options.as_ptr(),
                type_ptrs.as_mut_ptr(),
                names.as_mut_ptr(),
                columns.len() as ffi::idx_t,
                dst,
            );
            result_from_duckdb_error_data(error_data)
        }
    }

    unsafe fn next_chunk(&self) -> Result<Option<ffi::duckdb_data_chunk>> {
        unsafe {
            if self.exhausted.get() {
                return Ok(None);
            }

            // duckdb_fetch_chunk is the non-deprecated fetcher for both
            // materialized and streaming results. That matters for CALL, which
            // can produce a materialized result even after execute_streaming.
            let chunk = ffi::duckdb_fetch_chunk(self.result.as_handle());
            if chunk.is_null() {
                self.check_result_error()?;
                self.exhausted.set(true);
                return Ok(None);
            }

            if ffi::duckdb_data_chunk_get_size(chunk) == 0 {
                let mut chunk = chunk;
                ffi::duckdb_destroy_data_chunk(&mut chunk);
                self.check_result_error()?;
                self.exhausted.set(true);
                return Ok(None);
            }

            Ok(Some(chunk))
        }
    }

    unsafe fn check_result_error(&self) -> Result<()> {
        unsafe {
            let Some(message) = result_error_message(self.result.as_mut_ptr()) else {
                return Ok(());
            };
            Err(duckdb_failure_from_message(message))
        }
    }

    unsafe fn data_chunk_to_struct_array(&self, chunk: ffi::duckdb_data_chunk) -> Result<StructArray> {
        unsafe {
            let mut arrays = FFI_ArrowArray::empty();
            self.chunk_to_arrow(chunk, &mut arrays as *mut _ as *mut ffi::ArrowArray)?;

            let array_data = from_ffi(arrays, &self.arrow_schema)
                .map_err(|err| arrow_conversion_failure("Could not import DuckDB Arrow array", err))?;
            Ok(StructArray::from(array_data))
        }
    }

    unsafe fn chunk_to_arrow(&self, mut chunk: ffi::duckdb_data_chunk, dst: *mut ffi::ArrowArray) -> Result<()> {
        unsafe {
            let error_data = ffi::duckdb_data_chunk_to_arrow(self.arrow_options.as_ptr(), chunk, dst);
            ffi::duckdb_destroy_data_chunk(&mut chunk);
            result_from_duckdb_error_data(error_data)
        }
    }

    /// The struct type to cast a batch to before handing it to Polars, when
    /// the result has decimals of more than 38 digits.
    #[cfg(feature = "polars")]
    fn polars_narrowed_type(&self) -> Result<Option<&arrow::datatypes::DataType>> {
        if self.polars_narrowed_type.get().is_none() {
            let wide = arrow::datatypes::DataType::Struct(self.schema.fields().clone());
            let narrowed = narrow_decimal256_type(&wide)?;
            let _ = self.polars_narrowed_type.set((narrowed != wide).then_some(narrowed));
        }
        Ok(self
            .polars_narrowed_type
            .get()
            .expect("polars narrowed type cache was just initialized")
            .as_ref())
    }

    #[cfg(feature = "polars")]
    fn polars_arrow_field(&self) -> Result<&polars_arrow::datatypes::Field> {
        if self.polars_arrow_field.get().is_none() {
            let field = unsafe { self.load_polars_arrow_field()? };
            let _ = self.polars_arrow_field.set(field);
        }
        Ok(self
            .polars_arrow_field
            .get()
            .expect("polars Arrow field cache was just initialized"))
    }

    #[cfg(feature = "polars")]
    unsafe fn load_polars_arrow_field(&self) -> Result<polars_arrow::datatypes::Field> {
        unsafe {
            let mut schema = polars_arrow::ffi::ArrowSchema::empty();
            Self::export_arrow_schema(
                &self.arrow_options,
                &self.columns,
                &mut schema as *mut _ as *mut ffi::ArrowSchema,
            )?;

            polars_arrow::ffi::import_field_from_c(&schema)
                .map_err(|err| arrow_conversion_failure("Could not import Polars Arrow field", err))
        }
    }
}

/// Replaces every `Decimal256` in an Arrow type by the `Decimal128` of the
/// same scale. Polars decimals hold 38 digits; DuckDB exports a `DECIMAL` of 39
/// to 76 digits, and from Arrow format version 1.1 also `HUGEINT` and
/// `UHUGEINT`, as `Decimal256`.
#[cfg(feature = "polars")]
fn narrow_decimal256_type(data_type: &arrow::datatypes::DataType) -> Result<arrow::datatypes::DataType> {
    use arrow::datatypes::{DECIMAL128_MAX_PRECISION, DECIMAL128_MAX_SCALE, DataType, Field};

    let narrow_field = |field: &Arc<Field>| -> Result<Arc<Field>> {
        let narrowed = narrow_decimal256_type(field.data_type())?;
        Ok(Arc::new(field.as_ref().clone().with_data_type(narrowed)))
    };
    Ok(match data_type {
        DataType::Decimal256(_, scale) if *scale > DECIMAL128_MAX_SCALE => {
            return Err(duckdb_failure_from_message(format!(
                "Polars decimals hold at most 38 digits: cannot convert a {data_type} column, cast it in SQL"
            )));
        }
        DataType::Decimal256(width, scale) => DataType::Decimal128((*width).min(DECIMAL128_MAX_PRECISION), *scale),
        DataType::List(field) => DataType::List(narrow_field(field)?),
        DataType::LargeList(field) => DataType::LargeList(narrow_field(field)?),
        DataType::FixedSizeList(field, size) => DataType::FixedSizeList(narrow_field(field)?, *size),
        DataType::Map(field, sorted) => DataType::Map(narrow_field(field)?, *sorted),
        DataType::Struct(fields) => DataType::Struct(fields.iter().map(narrow_field).collect::<Result<_>>()?),
        other => other.clone(),
    })
}

pub(crate) unsafe fn logical_type_from_duckdb_column(
    ptr: ffi::duckdb_logical_type,
    idx: usize,
) -> Result<LogicalTypeHandle> {
    if ptr.is_null() {
        return Err(duckdb_failure_from_message(format!(
            "Could not retrieve logical type for result column at index {idx}"
        )));
    }
    Ok(unsafe { LogicalTypeHandle::new(ptr) })
}

pub(crate) fn reject_unsupported_result_logical_type(idx: usize, logical_type: &LogicalTypeHandle) -> Result<()> {
    if logical_type.contains_type_id(LogicalTypeId::Variant) {
        return Err(Error::FromSqlConversionFailure(
            idx,
            Type::Variant,
            "decoding Variant columns is not supported".into(),
        ));
    }

    Ok(())
}

/// Owned DuckDB result handle.
///
/// DuckDB advances result state while fetching chunks, so the raw result lives
/// in an `UnsafeCell` and fetch methods take `&self`. The owner is `Send` but
/// not `Sync`, which keeps that interior mutation single-threaded.
#[derive(Debug)]
struct DuckdbResultHandle {
    result: UnsafeCell<ffi::duckdb_result>,
}

impl DuckdbResultHandle {
    fn new(result: ffi::duckdb_result) -> Self {
        Self {
            result: UnsafeCell::new(result),
        }
    }

    fn as_handle(&self) -> ffi::duckdb_result {
        unsafe { *self.result.get() }
    }

    fn as_mut_ptr(&self) -> *mut ffi::duckdb_result {
        self.result.get()
    }
}

impl Drop for DuckdbResultHandle {
    fn drop(&mut self) {
        unsafe {
            ffi::duckdb_destroy_result(self.result.get());
        }
    }
}

#[derive(Debug)]
struct ArrowOptionsHandle {
    ptr: NonNull<ffi::_duckdb_arrow_options>,
}

impl ArrowOptionsHandle {
    /// Takes ownership of a DuckDB Arrow options handle for `result`.
    ///
    /// # Safety
    ///
    /// `result` must be a live DuckDB result. The returned handle owns the
    /// options pointer and destroys it on drop.
    unsafe fn new(result: &DuckdbResultHandle) -> Result<Self> {
        unsafe {
            let arrow_options = ffi::duckdb_result_get_arrow_options(result.as_mut_ptr());
            let Some(arrow_options) = NonNull::new(arrow_options) else {
                return Err(duckdb_failure_from_message(
                    "Could not retrieve Arrow options for result",
                ));
            };
            Ok(Self { ptr: arrow_options })
        }
    }

    fn as_ptr(&self) -> ffi::duckdb_arrow_options {
        self.ptr.as_ptr()
    }
}

impl Drop for ArrowOptionsHandle {
    fn drop(&mut self) {
        let mut arrow_options = self.ptr.as_ptr();
        unsafe {
            ffi::duckdb_destroy_arrow_options(&mut arrow_options);
        }
    }
}
