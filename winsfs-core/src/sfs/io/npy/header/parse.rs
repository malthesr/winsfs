use nom::{
    branch::alt,
    bytes::complete::{is_not, tag},
    character::complete::{one_of, space0, u64},
    combinator::{all_consuming, map, opt, verify},
    error::ParseError,
    multi::separated_list1,
    sequence::{delimited, pair, separated_pair, terminated, tuple},
    IResult, Parser,
};

use super::{Endian, ParseHeaderError, Type, TypeDescriptor};

/// An entry in the npy header literal dict.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(super) enum Entry {
    Descr(TypeDescriptor),
    FortranOrder(bool),
    Shape(Vec<usize>),
}

pub(super) fn parse_header_dict(input: &str) -> Result<Vec<Entry>, ParseHeaderError> {
    parse_dict(input)
        .map(|(_, entries)| entries)
        .map_err(|_| ParseHeaderError(input.to_string()))
}

fn parse_dict(input: &str) -> IResult<&str, Vec<Entry>> {
    delimited(
        tag("{").and(space0),
        separated_list1_opt(dict_sep, parse_entry),
        space0.and(tag("}")),
    )(input)
}

fn quote<'a: 'b, 'b>(quote: &'b str) -> impl FnMut(&'a str) -> IResult<&'a str, &'a str> + 'b {
    delimited(tag(quote), is_not(quote), tag(quote))
}

fn whitespace_sep<'a: 'b, 'b>(sep: &'b str) -> impl FnMut(&'a str) -> IResult<&'a str, ()> + 'b {
    map(tuple((space0, tag(sep), space0)), |_| ())
}

fn entry_sep(input: &str) -> IResult<&str, ()> {
    whitespace_sep(":")(input)
}

fn parse_string(input: &str) -> IResult<&str, &str> {
    alt((quote("'"), quote("\"")))(input)
}

fn parse_target_string<'a: 'b, 'b>(
    target: &'b str,
) -> impl FnMut(&'a str) -> IResult<&'a str, &'a str> + 'b {
    verify(parse_string, move |s: &str| s == target)
}

fn parse_bool(input: &str) -> IResult<&str, bool> {
    alt((map(tag("True"), |_| true), map(tag("False"), |_| false)))(input)
}

fn parse_fortran_order_entry(input: &str) -> IResult<&str, Entry> {
    map(
        separated_pair(parse_target_string("fortran_order"), entry_sep, parse_bool),
        |(_, bool)| Entry::FortranOrder(bool),
    )(input)
}

fn parse_endian(input: &str) -> IResult<&str, Endian> {
    alt((
        map(one_of("|<"), |_| Endian::Little),
        map(tag(">"), |_| Endian::Big),
    ))(input)
}

fn parse_type(input: &str) -> IResult<&str, Type> {
    alt((
        map(tag("f4"), |_| Type::F4),
        map(tag("f8"), |_| Type::F8),
        map(tag("u1"), |_| Type::U1),
        map(tag("u2"), |_| Type::U2),
        map(tag("u4"), |_| Type::U4),
        map(tag("u8"), |_| Type::U8),
        map(tag("i1"), |_| Type::I1),
        map(tag("i2"), |_| Type::I2),
        map(tag("i4"), |_| Type::I4),
        map(tag("i8"), |_| Type::I8),
    ))(input)
}

fn parse_type_descriptor(input: &str) -> IResult<&str, TypeDescriptor> {
    map(pair(parse_endian, parse_type), |(endian, ty)| {
        TypeDescriptor::new(endian, ty)
    })(input)
}

fn parse_descr_entry(input: &str) -> IResult<&str, Entry> {
    map(
        separated_pair(
            parse_target_string("descr"),
            entry_sep,
            parse_string.and_then(all_consuming(parse_type_descriptor)),
        ),
        |(_, type_descriptor)| Entry::Descr(type_descriptor),
    )(input)
}

fn parse_usize(input: &str) -> IResult<&str, usize> {
    map(u64, |x| x.try_into().expect("cannot convert u64 to usize"))(input)
}

fn shape_sep(input: &str) -> IResult<&str, ()> {
    whitespace_sep(",")(input)
}

pub fn separated_list1_opt<'a, O, O2, E, F, G>(
    sep: G,
    f: F,
) -> impl FnMut(&'a str) -> IResult<&'a str, Vec<O>, E>
where
    F: Parser<&'a str, O, E>,
    G: Parser<&'a str, O2, E> + Clone,
    E: ParseError<&'a str>,
{
    terminated(separated_list1(sep.clone(), f), opt(sep))
}

fn parse_usize_sequence(input: &str) -> IResult<&str, Vec<usize>> {
    separated_list1_opt(shape_sep, parse_usize)(input)
}

fn parse_shape(input: &str) -> IResult<&str, Vec<usize>> {
    delimited(tag("("), parse_usize_sequence, tag(")"))(input)
}

fn parse_shape_entry(input: &str) -> IResult<&str, Entry> {
    map(
        separated_pair(parse_target_string("shape"), entry_sep, parse_shape),
        |(_, shape)| Entry::Shape(shape),
    )(input)
}

fn parse_entry(input: &str) -> IResult<&str, Entry> {
    alt((
        parse_descr_entry,
        parse_fortran_order_entry,
        parse_shape_entry,
    ))(input)
}

fn dict_sep(input: &str) -> IResult<&str, ()> {
    whitespace_sep(",")(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_bool() {
        assert_eq!(parse_bool("True"), Ok(("", true)));
        assert_eq!(parse_bool("False"), Ok(("", false)));
        assert!(matches!(parse_bool("true"), Err(_)));
    }

    #[test]
    fn test_parse_string() {
        assert_eq!(parse_string("'foo'"), Ok(("", "foo")));
        assert_eq!(parse_string("\"bar\""), Ok(("", "bar")));
        assert!(matches!(parse_string("\"baz'"), Err(_)));
        assert_eq!(parse_string("'foo'bar"), Ok(("bar", "foo")));
    }

    #[test]
    fn test_parse_fortran_order_entry() {
        assert_eq!(
            parse_fortran_order_entry("'fortran_order': True"),
            Ok(("", Entry::FortranOrder(true)))
        );
        assert_eq!(
            parse_fortran_order_entry("'fortran_order'  :False"),
            Ok(("", Entry::FortranOrder(false)))
        );
    }

    #[test]
    fn test_parse_endian() {
        assert_eq!(parse_endian("<"), Ok(("", Endian::Little)));
        assert_eq!(parse_endian("|"), Ok(("", Endian::Little)));
        assert_eq!(parse_endian(">"), Ok(("", Endian::Big)));
        assert_eq!(parse_endian("<f8"), Ok(("f8", Endian::Little)));
        assert_eq!(parse_endian(">i1"), Ok(("i1", Endian::Big)));
    }

    #[test]
    fn test_parse_type_descriptor() {
        assert_eq!(
            parse_type_descriptor("<f8"),
            Ok(("", TypeDescriptor::new(Endian::Little, Type::F8)))
        );
        assert_eq!(
            parse_type_descriptor(">u2"),
            Ok(("", TypeDescriptor::new(Endian::Big, Type::U2)))
        );
        assert_eq!(
            parse_type_descriptor("|i4"),
            Ok(("", TypeDescriptor::new(Endian::Little, Type::I4)))
        );
    }

    #[test]
    fn test_parse_descr_entry() {
        assert_eq!(
            parse_descr_entry("'descr' : '>i2'"),
            Ok(("", Entry::Descr(TypeDescriptor::new(Endian::Big, Type::I2))))
        );
        assert_eq!(
            parse_descr_entry("\"descr\":\"<f4\""),
            Ok((
                "",
                Entry::Descr(TypeDescriptor::new(Endian::Little, Type::F4))
            ))
        );
    }

    #[test]
    fn test_parse_usize() {
        assert_eq!(parse_usize("1"), Ok(("", 1)));
        assert_eq!(parse_usize("22"), Ok(("", 22)));
        assert_eq!(parse_usize("123,456"), Ok((",456", 123)));
    }

    #[test]
    fn test_parse_usize_sequence() {
        assert_eq!(parse_usize_sequence("1"), Ok(("", vec![1])));
        assert_eq!(parse_usize_sequence("1,2"), Ok(("", vec![1, 2])));
        assert_eq!(parse_usize_sequence("1,2,34,"), Ok(("", vec![1, 2, 34])));
        assert_eq!(parse_usize_sequence("1,  2,3"), Ok(("", vec![1, 2, 3])));
        assert_eq!(
            parse_usize_sequence("123,  23,  13,  "),
            Ok(("", vec![123, 23, 13]))
        );
    }

    #[test]
    fn test_parse_shape_entry() {
        assert_eq!(
            parse_shape_entry("'shape': (1,)"),
            Ok(("", Entry::Shape(vec![1])))
        );
        assert_eq!(
            parse_shape_entry("'shape' : (1, 2)"),
            Ok(("", Entry::Shape(vec![1, 2])))
        );
        assert_eq!(
            parse_shape_entry("'shape' : (11, 22, 33, )"),
            Ok(("", Entry::Shape(vec![11, 22, 33])))
        );
    }

    #[test]
    fn test_parse_entry() {
        assert_eq!(
            parse_entry("'descr': '<f8'"),
            Ok((
                "",
                Entry::Descr(TypeDescriptor::new(Endian::Little, Type::F8)),
            ))
        );
        assert_eq!(
            parse_entry("'descr': '<i8'}"),
            Ok((
                "}",
                Entry::Descr(TypeDescriptor::new(Endian::Little, Type::I8)),
            ))
        );
        assert_eq!(
            parse_shape_entry("'shape': (100)"),
            Ok(("", Entry::Shape(vec![100])))
        );
    }

    #[test]
    fn test_parse_dict() {
        assert_eq!(
            parse_dict("{'descr': '<f8', 'shape': (15, 3), 'fortran_order': False}"),
            Ok((
                "",
                vec![
                    Entry::Descr(TypeDescriptor::new(Endian::Little, Type::F8)),
                    Entry::Shape(vec![15, 3]),
                    Entry::FortranOrder(false),
                ]
            ))
        );
        assert_eq!(
            parse_dict("{ 'fortran_order': True }"),
            Ok(("", vec![Entry::FortranOrder(true),]))
        )
    }
}
