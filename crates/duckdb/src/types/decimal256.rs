use std::{
    fmt,
    hash::{Hash, Hasher},
};

use arrow::datatypes::i256;

use crate::{
    Error, Result, ToSql, ffi,
    types::{FromSql, FromSqlError, FromSqlResult, ToSqlOutput, Value, ValueRef},
};

use super::{Decimal, DecimalError, TimeUnit, decimal::fmt_scaled_digits, to_duckdb_hugeint};

/// A DuckDB `DECIMAL(width, scale)` value of up to 76 digits.
///
/// This is the carrier of the decimals that do not fit in [`Decimal`]: DuckDB
/// returns a `DECIMAL` of 39 to 76 digits as an Arrow `Decimal256`, and
/// `value` stores its scaled integer payload as an Arrow [`i256`]. For
/// example, the sum of two `DECIMAL(38, 2)` values is a `DECIMAL(39, 2)`.
///
/// `Display` prints the exact digits. Equality and hashing follow
/// [`Decimal`]: the scale and the payload count, the declared width does not.
///
/// Decimals of more than 76 digits have no Arrow representation, so a query
/// that returns one fails with an error from DuckDB; cast such a column to
/// `VARCHAR` in SQL to read its digits.
#[derive(Copy, Clone, Debug)]
pub struct Decimal256 {
    width: u8,
    scale: u8,
    value: i256,
}

impl PartialEq for Decimal256 {
    fn eq(&self, other: &Self) -> bool {
        self.scale == other.scale && self.value == other.value
    }
}

impl Eq for Decimal256 {}

impl Hash for Decimal256 {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.scale.hash(state);
        self.value.hash(state);
    }
}

impl fmt::Display for Decimal256 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let digits = self.value.to_string();
        let raw = digits.strip_prefix('-').unwrap_or(&digits);
        fmt_scaled_digits(f, !self.value.is_negative(), raw, self.scale)
    }
}

impl Decimal256 {
    /// The widest decimal an Arrow `Decimal256` can carry.
    pub const MAX_WIDTH: u8 = 76;

    /// Builds a valid `DECIMAL(width, scale)` wrapper.
    ///
    /// Returns [`DecimalError`] when `width` is outside the `1..=76` range,
    /// `scale` is larger than `width`, or `value` has more digits than `width`
    /// allows.
    pub fn new(width: u8, scale: u8, value: i256) -> std::result::Result<Self, DecimalError> {
        if width == 0 || width > Self::MAX_WIDTH {
            return Err(DecimalError::InvalidWidth {
                width,
                max: Self::MAX_WIDTH,
            });
        }
        if scale > width {
            return Err(DecimalError::ScaleExceedsWidth { scale, width });
        }
        let digits = decimal256_digits(value);
        if digits > width {
            return Err(DecimalError::ValueExceedsWidth { digits, width });
        }
        Ok(Self { width, scale, value })
    }

    pub(crate) fn validate_signed_scale(width: u8, scale: i8) -> std::result::Result<u8, String> {
        let scale = u8::try_from(scale).map_err(|_| format!("negative decimal scale is not supported: {scale}"))?;
        Self::new(width, scale, i256::ZERO).map_err(|err| err.to_string())?;
        Ok(scale)
    }

    /// Builds a decimal value from trusted DuckDB output.
    pub(crate) fn from_chunk(width: u8, scale: u8, value: i256) -> Self {
        debug_assert!(
            Self::new(width, scale, value).is_ok(),
            "DuckDB produced an invalid decimal triple"
        );
        Self { width, scale, value }
    }

    /// Returns the declared precision.
    #[inline]
    pub const fn width(self) -> u8 {
        self.width
    }

    /// Returns the number of fractional digits.
    #[inline]
    pub const fn scale(self) -> u8 {
        self.scale
    }

    /// Returns the scaled integer payload.
    #[inline]
    pub const fn value(self) -> i256 {
        self.value
    }
}

/// Number of decimal digits of the magnitude of `value`, at least 1.
fn decimal256_digits(value: i256) -> u8 {
    let digits = value.to_string();
    // An i256 has at most 77 digits.
    digits.strip_prefix('-').unwrap_or(&digits).len() as u8
}

fn integer_decimal256(value: i256) -> Decimal256 {
    // Every 128-bit integer has at most 39 digits.
    Decimal256::from_chunk(decimal256_digits(value), 0, value)
}

impl From<Decimal> for Decimal256 {
    fn from(decimal: Decimal) -> Self {
        Self {
            width: decimal.width(),
            scale: decimal.scale(),
            value: i256::from_i128(decimal.value()),
        }
    }
}

impl TryFrom<Decimal256> for Decimal {
    type Error = DecimalError;

    /// Narrows to the 38-digit carrier when the value fits exactly.
    ///
    /// A declared width above 38 is replaced by the smallest width that holds
    /// the payload and the scale. Fails when the payload has more than 38
    /// digits or the scale is above 38: nothing is rounded.
    fn try_from(decimal: Decimal256) -> std::result::Result<Self, Self::Error> {
        let digits = decimal256_digits(decimal.value);
        let Some(value) = decimal.value.to_i128().filter(|_| digits <= Self::MAX_WIDTH) else {
            return Err(DecimalError::ValueExceedsWidth {
                digits,
                width: Self::MAX_WIDTH,
            });
        };
        if decimal.scale > Self::MAX_WIDTH {
            return Err(DecimalError::ScaleExceedsWidth {
                scale: decimal.scale,
                width: Self::MAX_WIDTH,
            });
        }
        let width = if decimal.width <= Self::MAX_WIDTH {
            decimal.width
        } else {
            digits.max(decimal.scale)
        };
        Self::new(width, decimal.scale, value)
    }
}

impl ToSql for Decimal256 {
    fn to_sql(&self) -> Result<ToSqlOutput<'_>> {
        Ok(ToSqlOutput::Owned(Value::Decimal256(*self)))
    }
}

impl FromSql for Decimal256 {
    fn column_result(value: ValueRef<'_>) -> FromSqlResult<Self> {
        let integer = match value {
            ValueRef::Decimal256(decimal) => return Ok(decimal),
            ValueRef::Decimal(decimal) => return Ok(decimal.into()),
            ValueRef::TinyInt(i) => i256::from(i),
            ValueRef::SmallInt(i) => i256::from(i),
            ValueRef::Int(i) => i256::from(i),
            ValueRef::BigInt(i) => i256::from(i),
            ValueRef::HugeInt(i) => i256::from(i),
            ValueRef::UHugeInt(i) => i256::from_parts(i, 0),
            ValueRef::UTinyInt(i) => i256::from(i64::from(i)),
            ValueRef::USmallInt(i) => i256::from(i64::from(i)),
            ValueRef::UInt(i) => i256::from(i64::from(i)),
            ValueRef::UBigInt(i) => i256::from(i128::from(i)),
            ValueRef::Timestamp(_, i) => i256::from(i),
            ValueRef::Date32(i) => i256::from(i),
            ValueRef::Time64(TimeUnit::Microsecond, i) => i256::from(i),
            _ => return Err(FromSqlError::InvalidType),
        };
        Ok(integer_decimal256(integer))
    }
}

fn conversion_error(message: String) -> FromSqlError {
    FromSqlError::Other(Box::new(std::io::Error::new(std::io::ErrorKind::InvalidData, message)))
}

/// Narrows a wide decimal for the conversions that go through [`Decimal`].
pub(super) fn narrow_decimal256(decimal: Decimal256) -> FromSqlResult<Decimal> {
    Decimal::try_from(decimal).map_err(|err| FromSqlError::Other(Box::new(err)))
}

/// Rounds a wide decimal to an integer like DuckDB's DECIMAL-to-INTEGER cast:
/// to nearest, with ties away from zero. Fails when the result needs more
/// than 128 bits.
pub(super) fn decimal256_to_rounded_i128(decimal: Decimal256) -> FromSqlResult<i128> {
    let out_of_range = || conversion_error(format!("DECIMAL value {decimal} is out of range for the integer type"));

    // 10^76 and twice a remainder below it both fit in an i256.
    let divisor = i256::from_i128(10)
        .checked_pow(u32::from(decimal.scale))
        .ok_or_else(out_of_range)?;
    let quotient = decimal.value.wrapping_div(divisor);
    let remainder = decimal.value.wrapping_rem(divisor);
    let twice = remainder.wrapping_abs().wrapping_add(remainder.wrapping_abs());
    let rounded = if twice < divisor {
        quotient
    } else if decimal.value.is_negative() {
        quotient.wrapping_sub(i256::ONE)
    } else {
        quotient.wrapping_add(i256::ONE)
    };
    rounded.to_i128().ok_or_else(out_of_range)
}

/// The floating point value nearest to a wide decimal, by parsing its exact
/// digits.
pub(super) fn decimal256_to_float<T>(decimal: Decimal256) -> FromSqlResult<T>
where
    T: std::str::FromStr,
{
    decimal
        .to_string()
        .parse::<T>()
        .map_err(|_| conversion_error(format!("DECIMAL value {decimal} cannot be converted to a float")))
}

/// Converts a wide decimal to the C API's 128-bit decimal struct.
///
/// The v1 C API has no value constructor for a decimal payload of more than
/// 128 bits, so those cannot be bound as parameters.
pub(crate) fn to_duckdb_decimal256(decimal: Decimal256) -> Result<ffi::duckdb_decimal> {
    let Some(value) = decimal.value.to_i128() else {
        return Err(Error::ToSqlConversionFailure(
            format!(
                "cannot bind DECIMAL({}, {}) value {decimal}: the DuckDB C API carries decimal parameters of at most \
                 128 bits; bind the digits as text and cast them in SQL, or insert through an Appender",
                decimal.width, decimal.scale
            )
            .into(),
        ));
    };
    Ok(ffi::duckdb_decimal {
        width: decimal.width,
        scale: decimal.scale,
        value: to_duckdb_hugeint(value),
    })
}

/// Encodes an integer as the BIGNUM blob DuckDB stores for the unscaled value
/// of a `DECIMAL` wider than 38 digits (`Bignum::SetHeader` and
/// `Bignum::FromByteArray` in DuckDB).
///
/// Layout: a three byte header holding the number of magnitude bytes with bit
/// 23 set, followed by the magnitude in big endian without leading zero
/// bytes (zero is a single `0x00`). A negative value stores the complement of
/// every byte, header included.
#[cfg_attr(not(feature = "vtab-arrow"), allow(dead_code))]
pub(crate) fn i256_to_bignum_blob(value: i256) -> Vec<u8> {
    let is_negative = value.is_negative();
    // Reinterpreting the bytes of `wrapping_abs` as unsigned is the magnitude,
    // including for `i256::MIN`.
    let magnitude = value.wrapping_abs().to_be_bytes();
    let first = magnitude
        .iter()
        .position(|byte| *byte != 0)
        .unwrap_or(magnitude.len() - 1);
    let data = &magnitude[first..];

    let header = data.len() as u32 | 0x0080_0000;
    let mut blob = Vec::with_capacity(3 + data.len());
    blob.extend_from_slice(&header.to_be_bytes()[1..]);
    blob.extend_from_slice(data);
    if is_negative {
        for byte in &mut blob {
            *byte = !*byte;
        }
    }
    blob
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::{Connection, params};

    fn wide_connection() -> Result<Connection> {
        let db = Connection::open_in_memory()?;
        // Decimal256 exists from Arrow format version 1.1.
        db.execute_batch("SET arrow_output_version = '1.1'")?;
        Ok(db)
    }

    /// Whether this DuckDB exports HUGEINT and UHUGEINT as Decimal256(39, 0)
    /// at Arrow format version 1.1, or still as Decimal128(38, 0).
    fn int128_exports_as_decimal256(db: &Connection) -> Result<bool> {
        use arrow::datatypes::DataType;

        let mut stmt = db.prepare("SELECT 1::HUGEINT, 1::UHUGEINT")?;
        let schema = stmt.query_arrow([])?.get_schema();
        let types = [schema.field(0).data_type().clone(), schema.field(1).data_type().clone()];
        if types == [DataType::Decimal256(39, 0), DataType::Decimal256(39, 0)] {
            Ok(true)
        } else {
            assert_eq!(types, [DataType::Decimal128(38, 0), DataType::Decimal128(38, 0)]);
            Ok(false)
        }
    }

    fn d256(width: u8, scale: u8, digits: &str) -> Decimal256 {
        Decimal256::new(width, scale, i256::from_string(digits).unwrap()).unwrap()
    }

    const NINES_38: &str = "99999999999999999999999999999999999999";
    const NINES_76: &str = "9999999999999999999999999999999999999999999999999999999999999999999999999999";

    #[test]
    fn decimal256_new_validates() {
        assert!(Decimal256::new(76, 76, i256::ZERO).is_ok());
        assert_eq!(
            Decimal256::new(77, 0, i256::ZERO).unwrap_err(),
            DecimalError::InvalidWidth { width: 77, max: 76 }
        );
        assert_eq!(
            Decimal256::new(0, 0, i256::ZERO).unwrap_err(),
            DecimalError::InvalidWidth { width: 0, max: 76 }
        );
        assert_eq!(
            Decimal256::new(40, 41, i256::ZERO).unwrap_err(),
            DecimalError::ScaleExceedsWidth { scale: 41, width: 40 }
        );
        assert_eq!(
            Decimal256::new(39, 0, i256::from_string(&format!("{NINES_38}99")).unwrap()).unwrap_err(),
            DecimalError::ValueExceedsWidth { digits: 40, width: 39 }
        );
        assert_eq!(
            Decimal256::new(76, 0, i256::MIN).unwrap_err(),
            DecimalError::ValueExceedsWidth { digits: 77, width: 76 }
        );
    }

    #[test]
    fn decimal256_display_is_exact() {
        assert_eq!(d256(76, 0, NINES_76).to_string(), NINES_76);
        assert_eq!(
            d256(76, 2, &format!("-{NINES_76}")).to_string(),
            format!("-{}.99", &NINES_76[..74])
        );
        assert_eq!(d256(40, 40, "-1").to_string(), format!("-0.{}1", "0".repeat(39)));
        assert_eq!(d256(39, 0, "0").to_string(), "0");
    }

    #[test]
    fn decimal256_identity_ignores_width() {
        assert_eq!(d256(39, 2, "150"), d256(76, 2, "150"));
        assert_ne!(d256(39, 2, "150"), d256(39, 1, "150"));
        assert_eq!(Decimal256::from(Decimal::new(5, 2, 150).unwrap()), d256(39, 2, "150"));
    }

    #[test]
    fn decimal256_narrows_only_when_exact() {
        let narrow = Decimal::try_from(d256(39, 2, "300")).unwrap();
        assert_eq!(narrow, Decimal::new(3, 2, 300).unwrap());
        assert_eq!(narrow.width(), 3);
        assert_eq!(Decimal::try_from(d256(20, 2, "300")).unwrap().width(), 20);
        assert_eq!(
            Decimal::try_from(d256(76, 0, NINES_38)).unwrap(),
            Decimal::new(38, 0, NINES_38.parse().unwrap()).unwrap()
        );

        assert_eq!(
            Decimal::try_from(d256(39, 0, &format!("1{}", "0".repeat(38)))).unwrap_err(),
            DecimalError::ValueExceedsWidth { digits: 39, width: 38 }
        );
        assert_eq!(
            Decimal::try_from(d256(76, 0, NINES_76)).unwrap_err(),
            DecimalError::ValueExceedsWidth { digits: 76, width: 38 }
        );
        assert_eq!(
            Decimal::try_from(d256(40, 39, "1")).unwrap_err(),
            DecimalError::ScaleExceedsWidth { scale: 39, width: 38 }
        );
    }

    #[test]
    fn bignum_blob_layout_matches_duckdb() {
        assert_eq!(i256_to_bignum_blob(i256::ZERO), [0x80, 0x00, 0x01, 0x00]);
        assert_eq!(i256_to_bignum_blob(i256::ONE), [0x80, 0x00, 0x01, 0x01]);
        assert_eq!(i256_to_bignum_blob(i256::MINUS_ONE), [0x7F, 0xFF, 0xFE, 0xFE]);
        assert_eq!(i256_to_bignum_blob(i256::from(256)), [0x80, 0x00, 0x02, 0x01, 0x00]);
        assert_eq!(i256_to_bignum_blob(i256::from(-256)), [0x7F, 0xFF, 0xFD, 0xFE, 0xFF]);
        let max = i256_to_bignum_blob(i256::MAX);
        assert_eq!(&max[..4], [0x80, 0x00, 0x20, 0x7F]);
        assert_eq!(max.len(), 35);
    }

    #[test]
    fn default_session_wide_arithmetic_reads_exactly() -> Result<()> {
        let db = wide_connection()?;

        let (ty, value): (String, Value) = db.query_row(
            "SELECT typeof(d), d FROM (SELECT 1.5::DECIMAL(38,2) + 1.5::DECIMAL(38,2) AS d)",
            [],
            |row| Ok((row.get(0)?, row.get(1)?)),
        )?;
        assert_eq!(ty, "DECIMAL(39,2)");
        let Value::Decimal256(decimal) = value else {
            panic!("expected a wide decimal, got {value:?}");
        };
        assert_eq!((decimal.width(), decimal.scale()), (39, 2));
        assert_eq!(decimal.value(), i256::from(300));
        assert_eq!(decimal.to_string(), "3.00");

        // A sum that no longer fits in 38 digits.
        let sum: Decimal256 = db.query_row(
            &format!("SELECT {NINES_38}::DECIMAL(38,0) + {NINES_38}::DECIMAL(38,0)"),
            [],
            |row| row.get(0),
        )?;
        assert_eq!(sum.to_string(), "199999999999999999999999999999999999998");
        Ok(())
    }

    #[test]
    fn wide_decimal_columns_read_exactly() -> Result<()> {
        let db = wide_connection()?;
        db.execute_batch(&format!(
            "CREATE TABLE wide (id INTEGER, d39 DECIMAL(39,0), d76 DECIMAL(76,10), dfrac DECIMAL(50,50));
             INSERT INTO wide VALUES
                (1, '{n39}', '{int76}.{frac76}', '0.{frac50}'),
                (2, '-{n39}', '-{int76}.{frac76}', '-0.{frac50}'),
                (3, 0, 0, 0),
                (4, NULL, NULL, NULL);",
            n39 = &NINES_76[..39],
            int76 = &NINES_76[..66],
            frac76 = "0123456789",
            frac50 = "1".repeat(50),
        ))?;

        let mut stmt = db.prepare("SELECT d39, d76, dfrac FROM wide ORDER BY id")?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get::<_, Option<Decimal256>>(0)?,
                    row.get::<_, Option<Decimal256>>(1)?,
                    row.get::<_, Option<Decimal256>>(2)?,
                ))
            })?
            .collect::<Result<Vec<_>>>()?;

        let strings = |idx: usize| {
            let (a, b, c) = rows[idx];
            [a, b, c].map(|d| d.map(|d| d.to_string()))
        };
        let n39 = &NINES_76[..39];
        let d76 = format!("{}.0123456789", &NINES_76[..66]);
        let frac = format!("0.{}", "1".repeat(50));
        assert_eq!(
            strings(0),
            [Some(n39.to_string()), Some(d76.clone()), Some(frac.clone())]
        );
        assert_eq!(
            strings(1),
            [
                Some(format!("-{n39}")),
                Some(format!("-{d76}")),
                Some(format!("-{frac}"))
            ]
        );
        assert_eq!(
            strings(2),
            [
                Some("0".to_string()),
                Some("0.0000000000".to_string()),
                Some(format!("0.{}", "0".repeat(50)))
            ]
        );
        assert_eq!(rows[3], (None, None, None));

        let (a, b, c) = rows[0];
        assert_eq!((a.unwrap().width(), a.unwrap().scale()), (39, 0));
        assert_eq!((b.unwrap().width(), b.unwrap().scale()), (76, 10));
        assert_eq!((c.unwrap().width(), c.unwrap().scale()), (50, 50));

        // The column is a Decimal256 for consumers of the Arrow batches too.
        let batches: Vec<_> = db.prepare("SELECT d76 FROM wide")?.query_arrow([])?.collect();
        assert_eq!(
            batches[0].schema().field(0).data_type(),
            &arrow::datatypes::DataType::Decimal256(76, 10)
        );
        Ok(())
    }

    #[test]
    fn decimals_beyond_arrow_fail_with_a_clean_error() -> Result<()> {
        let db = wide_connection()?;
        for width in [77, 255, 300] {
            let err = db
                .query_row(&format!("SELECT 1::DECIMAL({width},0)"), [], |row| {
                    row.get::<_, Decimal256>(0)
                })
                .unwrap_err();
            let message = err.to_string();
            assert!(
                message.contains("Arrow decimals support at most 76 digits"),
                "unexpected error for width {width}: {message}"
            );

            // The digits stay reachable through a VARCHAR cast.
            let digits = "7".repeat(width);
            let text: String =
                db.query_row(&format!("SELECT '-{digits}'::DECIMAL({width},0)::VARCHAR"), [], |row| {
                    row.get(0)
                })?;
            assert_eq!(text, format!("-{digits}"));
        }
        Ok(())
    }

    #[test]
    fn wide_decimal_converts_to_other_types_exactly_or_fails() -> Result<()> {
        let db = wide_connection()?;

        // Fits the 38-digit carrier.
        let narrow: Decimal = db.query_row("SELECT 1.5::DECIMAL(38,2) + 1.5::DECIMAL(38,2)", [], |row| row.get(0))?;
        assert_eq!(narrow, Decimal::new(3, 2, 300)?);
        let int: i64 = db.query_row("SELECT 2.5::DECIMAL(39,1)", [], |row| row.get(0))?;
        assert_eq!(int, 3);
        let int: i64 = db.query_row("SELECT -2.5::DECIMAL(39,1)", [], |row| row.get(0))?;
        assert_eq!(int, -3);
        let int: i64 = db.query_row("SELECT -2.4::DECIMAL(39,1)", [], |row| row.get(0))?;
        assert_eq!(int, -2);
        let float: f64 = db.query_row("SELECT -0.1::DECIMAL(60,30)", [], |row| row.get(0))?;
        assert_eq!(float, -0.1);

        // Does not fit: an error, never a rounded value.
        let sql = format!("SELECT {NINES_76}::DECIMAL(76,0)");
        let err = db.query_row(&sql, [], |row| row.get::<_, Decimal>(0)).unwrap_err();
        assert!(matches!(
            err,
            Error::FromSqlConversionFailure(0, crate::types::Type::Decimal, _)
        ));
        assert_eq!(
            err.to_string(),
            "Conversion error from type Decimal at index: 0, decimal value has 76 digits, exceeding width 38"
        );
        let err = db.query_row(&sql, [], |row| row.get::<_, i128>(0)).unwrap_err();
        assert!(matches!(
            err,
            Error::FromSqlConversionFailure(0, crate::types::Type::Decimal, _)
        ));
        let float: f64 = db.query_row(&sql, [], |row| row.get(0))?;
        assert_eq!(float, 1e76);
        Ok(())
    }

    #[cfg(feature = "rust_decimal")]
    #[test]
    fn rust_decimal_reads_wide_decimal_when_it_fits() -> Result<()> {
        let db = wide_connection()?;
        let read: rust_decimal::Decimal =
            db.query_row("SELECT 1.5::DECIMAL(38,2) + 1.5::DECIMAL(38,2)", [], |row| row.get(0))?;
        assert_eq!(read, rust_decimal::Decimal::from_i128_with_scale(300, 2));

        // rust_decimal holds 96 bits and a scale of at most 28.
        for sql in [
            format!("SELECT {NINES_76}::DECIMAL(76,0)"),
            "SELECT 1::DECIMAL(60,30)".to_string(),
            format!("SELECT {}::DECIMAL(39,0)", &NINES_76[..30]),
        ] {
            let err = db
                .query_row(&sql, [], |row| row.get::<_, rust_decimal::Decimal>(0))
                .unwrap_err();
            assert!(
                matches!(err, Error::FromSqlConversionFailure(0, crate::types::Type::Decimal, _)),
                "{sql}: {err:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn hugeint_extremes_read_exactly_at_arrow_1_1() -> Result<()> {
        let db = wide_connection()?;

        let mut stmt = db.prepare(
            "SELECT h, u, typeof(h), typeof(u) FROM (VALUES
                ((-170141183460469231731687303715884105727 - 1)::HUGEINT, 0::UHUGEINT),
                (170141183460469231731687303715884105727::HUGEINT, 340282366920938463463374607431768211455::UHUGEINT),
                (NULL, NULL)) t(h, u)",
        )?;
        let rows = stmt
            .query_map([], |row| {
                Ok((
                    row.get_ref(0)?.to_owned(),
                    row.get_ref(1)?.to_owned(),
                    row.get::<_, Option<i128>>(0)?,
                    row.get::<_, Option<u128>>(1)?,
                ))
            })?
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(
            rows,
            vec![
                (Value::HugeInt(i128::MIN), Value::UHugeInt(0), Some(i128::MIN), Some(0)),
                (
                    Value::HugeInt(i128::MAX),
                    Value::UHugeInt(u128::MAX),
                    Some(i128::MAX),
                    Some(u128::MAX)
                ),
                (Value::Null, Value::Null, None, None),
            ]
        );

        // An integer sum is a HUGEINT.
        let sum = db.query_row("SELECT sum(i) FROM range(10) t(i)", [], |row| {
            assert_eq!(row.get_ref(0)?, ValueRef::HugeInt(45));
            row.get::<_, i64>(0)
        })?;
        assert_eq!(sum, 45);

        // DECIMAL(39,0) has the same Arrow type as the 128-bit integers; the
        // result metadata keeps it a decimal.
        let value = db.query_row("SELECT 5::DECIMAL(39,0)", [], |row| {
            row.get_ref(0).map(|v| v.to_owned())
        })?;
        assert_eq!(value, Value::Decimal256(d256(39, 0, "5")));
        let Value::Decimal256(decimal) = value else {
            unreachable!()
        };
        assert_eq!(decimal.width(), 39);
        Ok(())
    }

    #[test]
    fn nested_128_bit_integers_read_exactly_at_arrow_1_1() -> Result<()> {
        let db = wide_connection()?;
        let list = db.query_row(
            "SELECT [170141183460469231731687303715884105727::HUGEINT, -1::HUGEINT, NULL]",
            [],
            |row| row.get_ref(0).map(|v| v.to_owned()),
        )?;
        assert_eq!(
            list,
            Value::List(vec![Value::HugeInt(i128::MAX), Value::HugeInt(-1), Value::Null])
        );

        // Without metadata for the child, a value above the signed range is
        // still exact once DuckDB exports the 128-bit integers as Decimal256.
        // (Before that a nested UHUGEINT of 2^127 or more arrives as a
        // negative Decimal128.)
        if int128_exports_as_decimal256(&db)? {
            let list = db.query_row(
                "SELECT [340282366920938463463374607431768211455::UHUGEINT, 1::UHUGEINT]",
                [],
                |row| row.get_ref(0).map(|v| v.to_owned()),
            )?;
            assert_eq!(list, Value::List(vec![Value::UHugeInt(u128::MAX), Value::HugeInt(1)]));
        }

        let list = db.query_row(
            &format!(
                "SELECT [-{n39}::DECIMAL(39,0)], [1.5::DECIMAL(40,1)]",
                n39 = &NINES_76[..39]
            ),
            [],
            |row| Ok((row.get_ref(0)?.to_owned(), row.get_ref(1)?.to_owned())),
        )?;
        assert_eq!(
            list,
            (
                Value::List(vec![Value::Decimal256(d256(39, 0, &format!("-{}", &NINES_76[..39])))]),
                Value::List(vec![Value::Decimal256(d256(40, 1, "15"))]),
            )
        );
        Ok(())
    }

    #[test]
    fn wide_decimal_parameters_bind_exactly() -> Result<()> {
        let db = wide_connection()?;

        let value = d256(50, 2, &format!("-{NINES_38}"));
        let (ty, read): (String, Decimal256) = db.query_row("SELECT typeof(?), ?", params![value, value], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?;
        assert_eq!(ty, "DECIMAL(50,2)");
        assert_eq!(read, value);
        assert_eq!(read.width(), 50);

        // Narrow widths go through the same carrier.
        let narrow = d256(10, 2, "-12345");
        let (ty, read): (String, Decimal) = db.query_row("SELECT typeof(?), ?", params![narrow, narrow], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?;
        assert_eq!(ty, "DECIMAL(10,2)");
        assert_eq!(read, Decimal::new(10, 2, -12345)?);

        // Nested values.
        let list = Value::List(vec![Value::Decimal256(value), Value::Null]);
        let (ty, read): (String, Value) = db.query_row("SELECT typeof(?), ?", params![list, list], |row| {
            Ok((row.get(0)?, row.get(1)?))
        })?;
        assert_eq!(ty, "DECIMAL(50,2)[]");
        assert_eq!(read, list);

        // More than 128 bits cannot go through the C API's decimal struct.
        let too_wide = d256(76, 0, NINES_76);
        let err = db
            .query_row("SELECT ?", params![too_wide], |row| row.get::<_, Decimal256>(0))
            .unwrap_err();
        assert!(err.to_string().contains("at most 128 bits"), "unexpected error: {err}");
        // Text plus a cast in SQL is exact.
        let read: Decimal256 = db.query_row("SELECT ?::DECIMAL(76,0)", params![too_wide.to_string()], |row| {
            row.get(0)
        })?;
        assert_eq!(read, too_wide);
        Ok(())
    }

    #[test]
    fn wide_decimals_append_exactly() -> Result<()> {
        let db = wide_connection()?;
        db.execute_batch("CREATE TABLE wide (d DECIMAL(76,10), n DECIMAL(20,2))")?;

        let max = d256(76, 10, NINES_76);
        let min = d256(76, 10, &format!("-{NINES_76}"));
        let small = d256(76, 10, "-15000000000");
        {
            let mut appender = db.appender("wide")?;
            appender.append_row(params![max, d256(20, 2, "125")])?;
            appender.append_row(params![Value::Decimal256(min), Value::Decimal256(d256(40, 2, "-125"))])?;
            appender.append_row(params![small, None::<Decimal256>])?;
            appender.append_row(params![None::<Decimal256>, Decimal::new(20, 2, 7)?])?;
        }

        let rows = db
            .prepare("SELECT d, n FROM wide ORDER BY d NULLS LAST")?
            .query_map([], |row| {
                Ok((row.get::<_, Option<Decimal256>>(0)?, row.get::<_, Option<Decimal>>(1)?))
            })?
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(
            rows,
            vec![
                (Some(min), Some(Decimal::new(20, 2, -125)?)),
                (Some(small), None),
                (Some(max), Some(Decimal::new(20, 2, 125)?)),
                (None, Some(Decimal::new(20, 2, 7)?)),
            ]
        );
        Ok(())
    }
}
