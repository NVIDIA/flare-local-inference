use std::collections::HashMap;

use serde_json::Value as Json;

use super::{CompiledTemplate, Error, Result};

#[derive(Clone, Copy, Debug)]
pub(super) enum Instruction {
    Text(u32),
    Constant(u32),
    Load(u32),
    Store(u32),
    StoreAttr(u32, u32),
    AppendAttr(u32, u32),
    Attr(u32),
    Index,
    Slice,
    List(u32),
    Map(u32),
    Unary(Unary),
    Binary(Binary),
    Test(u32),
    Filter(u32),
    Call(u32),
    Method(u32),
    Emit,
    Join(u32, bool),
    EmitJoined(u32, bool),
    Drop,
    Jump(u32),
    JumpFalse(u32),
    And(u32),
    Or(u32),
    For(u32),
    Next(u32),
    EndLoop,
    Break(u32),
    Continue(u32),
    Macro(u32, u32),
    Return,
    Capture,
    EndCapture,
    Mark,
    MarkPosition,
}

#[derive(Clone, Copy, Debug)]
pub(super) enum Unary {
    Not,
    Neg,
    Pos,
}
#[derive(Clone, Copy, Debug)]
pub(super) enum Binary {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Rem,
    Pow,
    Concat,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    In,
    NotIn,
}

pub(super) struct Call {
    pub name: u32,
    pub positional: usize,
    pub keywords: Vec<u32>,
}
pub(super) struct Loop {
    pub variables: Vec<u32>,
    pub filter: Option<u32>,
    pub body: u32,
    pub empty: u32,
    pub end: u32,
    pub next: u32,
}
pub(super) struct Macro {
    pub parameters: Vec<u32>,
    pub entry: u32,
    pub defaults: Vec<(u32, u32)>,
    pub captures: Vec<u32>,
}

#[derive(Debug, Clone, PartialEq)]
enum Token {
    Name(String),
    String(String),
    Number(String),
    Op(String),
    End,
}

fn lex(source: &str) -> Result<Vec<Token>> {
    let mut tokens = Vec::new();
    let mut chars = source.char_indices().peekable();
    while let Some((start, c)) = chars.next() {
        if c.is_whitespace() {
            continue;
        }
        if c == '\'' || c == '"' {
            let mut value = String::new();
            let mut closed = false;
            while let Some((_, ch)) = chars.next() {
                if ch == c {
                    closed = true;
                    break;
                }
                if ch != '\\' {
                    value.push(ch);
                    continue;
                }
                let (_, escaped) = chars
                    .next()
                    .ok_or_else(|| Error::new("unterminated string escape"))?;
                match escaped {
                    'n' => value.push('\n'),
                    'r' => value.push('\r'),
                    't' => value.push('\t'),
                    'b' => value.push('\u{8}'),
                    'f' => value.push('\u{c}'),
                    '\\' | '\'' | '"' => value.push(escaped),
                    'x' | 'u' | 'U' => {
                        let width = match escaped {
                            'x' => 2,
                            'u' => 4,
                            _ => 8,
                        };
                        let mut code = 0;
                        for _ in 0..width {
                            code = code * 16
                                + chars
                                    .next()
                                    .and_then(|(_, c)| c.to_digit(16))
                                    .ok_or_else(|| Error::new("invalid Unicode escape"))?;
                        }
                        value.push(
                            char::from_u32(code)
                                .ok_or_else(|| Error::new("invalid Unicode scalar"))?,
                        );
                    }
                    other => {
                        value.push('\\');
                        value.push(other);
                    }
                }
            }
            if !closed {
                return Err(Error::new("unterminated string"));
            }
            tokens.push(Token::String(value));
        } else if c.is_ascii_digit() {
            let mut end = start + 1;
            while let Some(&(i, ch)) = chars.peek() {
                if ch.is_ascii_digit()
                    || ch == '.' && source.as_bytes().get(i + 1).is_some_and(u8::is_ascii_digit)
                {
                    chars.next();
                    end = i + ch.len_utf8();
                } else {
                    break;
                }
            }
            if chars.peek().is_some_and(|(_, c)| matches!(c, 'e' | 'E')) {
                end = chars.next().unwrap().0 + 1;
                if chars.peek().is_some_and(|(_, c)| matches!(c, '+' | '-')) {
                    end = chars.next().unwrap().0 + 1;
                }
                while chars.peek().is_some_and(|(_, c)| c.is_ascii_digit()) {
                    end = chars.next().unwrap().0 + 1;
                }
            }
            tokens.push(Token::Number(source[start..end].into()));
        } else if c == '_' || c.is_alphabetic() {
            let mut end = start + c.len_utf8();
            while let Some(&(i, ch)) = chars.peek() {
                if ch != '_' && !ch.is_alphanumeric() {
                    break;
                }
                chars.next();
                end = i + ch.len_utf8();
            }
            tokens.push(Token::Name(source[start..end].into()));
        } else {
            let mut op = c.to_string();
            if let Some(&(_, next)) = chars.peek() {
                if matches!(
                    (c, next),
                    ('=', '=') | ('!', '=') | ('<', '=') | ('>', '=') | ('/', '/') | ('*', '*')
                ) {
                    op.push(next);
                    chars.next();
                }
            }
            tokens.push(Token::Op(op));
        }
    }
    tokens.push(Token::End);
    Ok(tokens)
}

#[derive(Clone)]
enum Expr {
    Constant(Json),
    Name(String),
    Attr(Box<Expr>, String),
    Index(Box<Expr>, Box<Expr>),
    Slice(Box<Expr>, Box<Expr>, Box<Expr>, Box<Expr>),
    List(Vec<Expr>),
    Map(Vec<(Expr, Expr)>),
    Unary(Unary, Box<Expr>),
    Binary(Binary, Box<Expr>, Box<Expr>),
    And(Box<Expr>, Box<Expr>),
    Or(Box<Expr>, Box<Expr>),
    If(Box<Expr>, Box<Expr>, Box<Expr>),
    Call(Box<Expr>, Args),
    Filter(Box<Expr>, String, Args),
    Test(Box<Expr>, String, Args),
}
#[derive(Clone, Default)]
struct Args {
    positional: Vec<Expr>,
    keywords: Vec<(String, Expr)>,
}

struct Parser {
    tokens: Vec<Token>,
    at: usize,
    depth: usize,
}
impl Parser {
    fn new(s: &str) -> Result<Self> {
        Ok(Self {
            tokens: lex(s)?,
            at: 0,
            depth: 0,
        })
    }
    fn peek(&self) -> &Token {
        &self.tokens[self.at]
    }
    fn is(&self, s: &str) -> bool {
        matches!(self.peek(),Token::Name(v)|Token::Op(v) if v==s)
    }
    fn eat(&mut self, s: &str) -> bool {
        if self.is(s) {
            self.at += 1;
            true
        } else {
            false
        }
    }
    fn require(&mut self, s: &str) -> Result<()> {
        if self.eat(s) {
            Ok(())
        } else {
            Err(Error::new(format!("expected {s}, found {:?}", self.peek())))
        }
    }
    fn name(&mut self) -> Result<String> {
        if let Token::Name(s) = self.peek().clone() {
            self.at += 1;
            Ok(s)
        } else {
            Err(Error::new("expected identifier"))
        }
    }
    fn finish(&self) -> Result<()> {
        if matches!(self.peek(), Token::End) {
            Ok(())
        } else {
            Err(Error::new(format!("unexpected token {:?}", self.peek())))
        }
    }
    fn args(&mut self) -> Result<Args> {
        let mut args = Args::default();
        if self.eat(")") {
            return Ok(args);
        }
        loop {
            if let Token::Name(name) = self.peek().clone() {
                if self.tokens.get(self.at + 1) == Some(&Token::Op("=".into())) {
                    self.at += 2;
                    args.keywords.push((name, self.expression(0)?));
                } else {
                    args.positional.push(self.expression(0)?);
                }
            } else {
                args.positional.push(self.expression(0)?);
            }
            if self.eat(")") {
                break;
            }
            self.require(",")?;
            if self.eat(")") {
                break;
            }
        }
        Ok(args)
    }
    fn expression(&mut self, min: u8) -> Result<Expr> {
        self.depth += 1;
        if self.depth > 128 {
            return Err(Error::new("expression nesting limit exceeded"));
        }
        let mut left = if self.eat("not") {
            Expr::Unary(Unary::Not, Box::new(self.expression(3)?))
        } else if self.eat("-") {
            Expr::Unary(Unary::Neg, Box::new(self.expression(9)?))
        } else if self.eat("+") {
            Expr::Unary(Unary::Pos, Box::new(self.expression(9)?))
        } else {
            self.primary()?
        };
        loop {
            if min <= 10 && self.eat(".") {
                let token = self.peek().clone();
                self.at += 1;
                left = match token {
                    Token::Name(name) => Expr::Attr(Box::new(left), name),
                    Token::Number(n) => Expr::Index(
                        Box::new(left),
                        Box::new(Expr::Constant(
                            n.parse::<i64>()
                                .map(Json::from)
                                .map_err(|_| Error::new("invalid numeric attribute"))?,
                        )),
                    ),
                    _ => return Err(Error::new("expected attribute name")),
                };
                continue;
            }
            if min <= 10 && self.eat("[") {
                let first = if self.is(":") {
                    Expr::Constant(Json::Null)
                } else {
                    self.expression(0)?
                };
                if self.eat(":") {
                    let end = if self.is(":") || self.is("]") {
                        Expr::Constant(Json::Null)
                    } else {
                        self.expression(0)?
                    };
                    let step = if self.eat(":") && !self.is("]") {
                        self.expression(0)?
                    } else {
                        Expr::Constant(Json::Null)
                    };
                    left = Expr::Slice(
                        Box::new(left),
                        Box::new(first),
                        Box::new(end),
                        Box::new(step),
                    );
                } else {
                    left = Expr::Index(Box::new(left), Box::new(first));
                }
                self.require("]")?;
                continue;
            }
            if min <= 10 && self.eat("(") {
                left = Expr::Call(Box::new(left), self.args()?);
                continue;
            }
            if min <= 10 && self.eat("|") {
                let name = self.name()?;
                let args = if self.eat("(") {
                    self.args()?
                } else {
                    Args::default()
                };
                left = Expr::Filter(Box::new(left), name, args);
                continue;
            }
            if min <= 4 && self.eat("is") {
                let negate = self.eat("not");
                let name = self.name()?;
                let args = if self.eat("(") {
                    self.args()?
                } else {
                    Args::default()
                };
                left = Expr::Test(Box::new(left), name, args);
                if negate {
                    left = Expr::Unary(Unary::Not, Box::new(left));
                }
                continue;
            }
            if min == 0 && self.eat("if") {
                let condition = self.expression(1)?;
                let otherwise = if self.eat("else") {
                    self.expression(0)?
                } else {
                    Expr::Name("undefined".into())
                };
                left = Expr::If(Box::new(condition), Box::new(left), Box::new(otherwise));
                continue;
            }
            let op = match self.peek() {
                Token::Name(v) | Token::Op(v) => v.as_str(),
                _ => break,
            };
            let (precedence, binary) = match op {
                "or" => (1, None),
                "and" => (2, None),
                "==" => (4, Some(Binary::Eq)),
                "!=" => (4, Some(Binary::Ne)),
                "<" => (4, Some(Binary::Lt)),
                "<=" => (4, Some(Binary::Le)),
                ">" => (4, Some(Binary::Gt)),
                ">=" => (4, Some(Binary::Ge)),
                "in" => (4, Some(Binary::In)),
                "not" if self.tokens.get(self.at + 1) == Some(&Token::Name("in".into())) => {
                    (4, Some(Binary::NotIn))
                }
                "~" => (5, Some(Binary::Concat)),
                "+" => (6, Some(Binary::Add)),
                "-" => (6, Some(Binary::Sub)),
                "*" => (7, Some(Binary::Mul)),
                "/" => (7, Some(Binary::Div)),
                "//" => (7, Some(Binary::FloorDiv)),
                "%" => (7, Some(Binary::Rem)),
                "**" => (8, Some(Binary::Pow)),
                _ => break,
            };
            if precedence < min {
                break;
            }
            self.at += if matches!(binary, Some(Binary::NotIn)) {
                2
            } else {
                1
            };
            let right = Box::new(self.expression(precedence + 1)?);
            left = if let Some(binary) = binary {
                Expr::Binary(binary, Box::new(left), right)
            } else if precedence == 1 {
                Expr::Or(Box::new(left), right)
            } else {
                Expr::And(Box::new(left), right)
            };
        }
        self.depth -= 1;
        Ok(left)
    }
    fn primary(&mut self) -> Result<Expr> {
        let token = self.peek().clone();
        self.at += 1;
        match token {
            Token::String(s) => Ok(Expr::Constant(s.into())),
            Token::Number(n) => Ok(Expr::Constant(
                serde_json::from_str(&n).map_err(|_| Error::new("invalid number"))?,
            )),
            Token::Name(n) => Ok(match n.as_str() {
                "true" | "True" => Expr::Constant(true.into()),
                "false" | "False" => Expr::Constant(false.into()),
                "none" | "None" => Expr::Constant(Json::Null),
                _ => Expr::Name(n),
            }),
            Token::Op(op) if op == "[" || op == "(" => {
                let end = if op == "[" { "]" } else { ")" };
                let mut items = Vec::new();
                let mut comma = false;
                if !self.eat(end) {
                    loop {
                        items.push(self.expression(0)?);
                        if self.eat(end) {
                            break;
                        }
                        self.require(",")?;
                        comma = true;
                        if self.eat(end) {
                            break;
                        }
                    }
                }
                if op == "(" && items.len() == 1 && !comma {
                    Ok(items.pop().unwrap())
                } else {
                    Ok(Expr::List(items))
                }
            }
            Token::Op(op) if op == "{" => {
                let mut pairs = Vec::new();
                if !self.eat("}") {
                    loop {
                        let key = self.expression(0)?;
                        self.require(":")?;
                        let value = self.expression(0)?;
                        pairs.push((key, value));
                        if self.eat("}") {
                            break;
                        }
                        self.require(",")?;
                        if self.eat("}") {
                            break;
                        }
                    }
                }
                Ok(Expr::Map(pairs))
            }
            _ => Err(Error::new(format!("expected expression, found {token:?}"))),
        }
    }
}

enum PartKind {
    Text(String),
    Expression(String),
    Statement(String),
}
struct Part {
    kind: PartKind,
    span: std::ops::Range<usize>,
}

fn parts(source: &str) -> Result<Vec<Part>> {
    fn without_final_newline(s: &str) -> &str {
        s.strip_suffix("\r\n")
            .or_else(|| s.strip_suffix('\n'))
            .unwrap_or(s)
    }
    let mut parts = Vec::new();
    let mut at = 0;
    let mut trim = false;
    while at < source.len() {
        let next = source[at..].find('{').map(|n| at + n);
        let Some(mut start) = next else {
            let text = if trim {
                source[at..].trim_start()
            } else {
                &source[at..]
            };
            parts.push(Part {
                kind: PartKind::Text(without_final_newline(text).into()),
                span: at..source.len(),
            });
            break;
        };
        while !matches!(source.as_bytes().get(start + 1), Some(b'{' | b'%' | b'#')) {
            if let Some(n) = source[start + 1..].find('{') {
                start += n + 1;
            } else {
                start = source.len();
                break;
            }
        }
        if start == source.len() {
            parts.push(Part {
                kind: PartKind::Text(without_final_newline(&source[at..]).into()),
                span: at..start,
            });
            break;
        }
        let kind = source.as_bytes()[start + 1];
        let left = source.as_bytes().get(start + 2) == Some(&b'-');
        let preserve = source.as_bytes().get(start + 2) == Some(&b'+');
        let mut text = &source[at..start];
        if trim {
            text = text.trim_start();
        }
        if left {
            text = text.trim_end();
        } else if kind != b'{' && !preserve {
            let line = text.rsplit_once('\n').map_or(text, |(_, s)| s);
            if line.chars().all(|c| matches!(c, ' ' | '\t')) {
                text = &text[..text.len() - line.len()];
            }
        }
        if !text.is_empty() {
            parts.push(Part {
                kind: PartKind::Text(text.into()),
                span: at..start,
            });
        }
        let body_start = start + 2 + usize::from(left || preserve);
        let delimiter = match kind {
            b'{' => "}}",
            b'%' => "%}",
            _ => "#}",
        };
        let mut quote = None;
        let mut escaped = false;
        let mut end = None;
        for (i, c) in source[body_start..].char_indices() {
            let i = i + body_start;
            if kind != b'#' {
                if escaped {
                    escaped = false;
                    continue;
                }
                if let Some(q) = quote {
                    if c == '\\' {
                        escaped = true;
                    } else if c == q {
                        quote = None;
                    }
                    continue;
                }
                if c == '\'' || c == '"' {
                    quote = Some(c);
                    continue;
                }
            }
            if source[i..].starts_with(delimiter) {
                end = Some(i);
                break;
            }
        }
        let end = end.ok_or_else(|| Error {
            message: "unclosed template tag".into(),
            offset: start,
        })?;
        trim = end > body_start && source.as_bytes().get(end - 1) == Some(&b'-');
        let keep = end > body_start && source.as_bytes().get(end - 1) == Some(&b'+');
        let body = &source[body_start..end - usize::from(trim || keep)];
        at = end + 2;
        if kind != b'{' && !trim && !keep {
            if source[at..].starts_with("\r\n") {
                at += 2;
            } else if source[at..].starts_with('\n') {
                at += 1;
            }
        }
        if kind != b'#' {
            parts.push(Part {
                kind: if kind == b'{' {
                    PartKind::Expression(body.into())
                } else {
                    PartKind::Statement(body.trim().into())
                },
                span: start..end + 2,
            });
        }
    }
    Ok(parts)
}

struct Compiler {
    template: CompiledTemplate,
    symbols: HashMap<String, u32>,
    parts: Vec<Part>,
    at: usize,
    offset: usize,
    captures: usize,
    loops: Vec<u32>,
    depth: usize,
}

pub(super) fn compile(source: &str) -> Result<CompiledTemplate> {
    let mut compiler = Compiler {
        template: CompiledTemplate {
            code: Vec::new(),
            constants: Vec::new(),
            names: Vec::new(),
            calls: Vec::new(),
            loops: Vec::new(),
            macros: Vec::new(),
            spans: Vec::new(),
            annotations: Vec::new(),
            loop_symbol: 0,
        },
        symbols: HashMap::new(),
        parts: parts(source)?,
        at: 0,
        offset: 0,
        captures: 0,
        loops: Vec::new(),
        depth: 0,
    };
    compiler.template.loop_symbol = compiler.symbol("loop");
    compiler.body(&[]).map_err(|mut e| {
        if e.offset == 0 {
            e.offset = compiler.offset;
        }
        e
    })?;
    Ok(compiler.template)
}

impl Compiler {
    fn joined<'e>(e: &'e Expr) -> Option<(bool, Vec<&'e Expr>)> {
        fn collect<'e>(e: &'e Expr, strict: bool, out: &mut Vec<&'e Expr>) {
            if let Expr::Binary(op, a, b) = e {
                if matches!((op, strict), (Binary::Add, true) | (Binary::Concat, false)) {
                    collect(a, strict, out);
                    collect(b, strict, out);
                    return;
                }
            }
            out.push(e);
        }
        let strict = match e {
            Expr::Binary(Binary::Add, ..) => true,
            Expr::Binary(Binary::Concat, ..) => false,
            _ => return None,
        };
        let mut items = Vec::new();
        collect(e, strict, &mut items);
        if strict
            && !items
                .iter()
                .any(|e| matches!(e, Expr::Constant(Json::String(_))))
        {
            return None;
        }
        Some((strict, items))
    }
    fn symbol(&mut self, s: &str) -> u32 {
        if let Some(&v) = self.symbols.get(s) {
            v
        } else {
            let v = self.template.names.len() as u32;
            self.template.names.push(s.into());
            self.symbols.insert(s.into(), v);
            v
        }
    }
    fn emit(&mut self, op: Instruction) -> usize {
        let i = self.template.code.len();
        self.template.code.push(op);
        self.template.spans.push(self.offset);
        i
    }
    fn pc(&self) -> u32 {
        self.template.code.len() as u32
    }
    fn constant(&mut self, v: Json) -> u32 {
        if let Some(i) = self.template.constants.iter().position(|c| c == &v) {
            return i as u32;
        }
        let i = self.template.constants.len() as u32;
        self.template.constants.push(v);
        i
    }
    fn jump(&mut self, i: usize, target: u32) {
        match &mut self.template.code[i] {
            Instruction::Jump(v)
            | Instruction::JumpFalse(v)
            | Instruction::And(v)
            | Instruction::Or(v) => *v = target,
            _ => unreachable!(),
        }
    }
    fn call_args(&mut self, name: &str, args: &Args) -> Result<u32> {
        for e in &args.positional {
            self.expr(e)?;
        }
        for (_, e) in &args.keywords {
            self.expr(e)?;
        }
        let keywords = args.keywords.iter().map(|(s, _)| self.symbol(s)).collect();
        let name = self.symbol(name);
        let i = self.template.calls.len() as u32;
        self.template.calls.push(Call {
            name,
            positional: args.positional.len(),
            keywords,
        });
        Ok(i)
    }
    fn expr(&mut self, e: &Expr) -> Result<()> {
        use Instruction as I;
        match e {
            Expr::Constant(v) => {
                let v = self.constant(v.clone());
                self.emit(I::Constant(v));
            }
            Expr::Name(n) => {
                let n = self.symbol(n);
                self.emit(I::Load(n));
            }
            Expr::Attr(v, n) => {
                self.expr(v)?;
                let n = self.symbol(n);
                self.emit(I::Attr(n));
            }
            Expr::Index(v, k) => {
                self.expr(v)?;
                self.expr(k)?;
                self.emit(I::Index);
            }
            Expr::Slice(v, a, b, c) => {
                for v in [v, a, b, c] {
                    self.expr(v)?;
                }
                self.emit(I::Slice);
            }
            Expr::List(items) => {
                for v in items {
                    self.expr(v)?;
                }
                self.emit(I::List(items.len() as u32));
            }
            Expr::Map(items) => {
                for (k, v) in items {
                    self.expr(k)?;
                    self.expr(v)?;
                }
                self.emit(I::Map(items.len() as u32));
            }
            Expr::Unary(op, v) => {
                self.expr(v)?;
                self.emit(I::Unary(*op));
            }
            Expr::Binary(op, a, b) => {
                if let Some((strict, items)) = Self::joined(e) {
                    for item in &items {
                        self.expr(item)?;
                    }
                    self.emit(I::Join(items.len() as u32, strict));
                } else {
                    self.expr(a)?;
                    self.expr(b)?;
                    self.emit(I::Binary(*op));
                }
            }
            Expr::And(a, b) | Expr::Or(a, b) => {
                self.expr(a)?;
                let i = self.emit(if matches!(e, Expr::And(..)) {
                    I::And(0)
                } else {
                    I::Or(0)
                });
                self.expr(b)?;
                self.jump(i, self.pc());
            }
            Expr::If(c, a, b) => {
                self.expr(c)?;
                let j = self.emit(I::JumpFalse(0));
                self.expr(a)?;
                let k = self.emit(I::Jump(0));
                self.jump(j, self.pc());
                self.expr(b)?;
                self.jump(k, self.pc());
            }
            Expr::Call(f, args) => match f.as_ref() {
                Expr::Attr(v, n) => {
                    self.expr(v)?;
                    let i = self.call_args(n, args)?;
                    self.emit(I::Method(i));
                }
                Expr::Name(n) => {
                    let i = self.call_args(n, args)?;
                    self.emit(I::Call(i));
                }
                _ => return Err(Error::new("call target must be a function or method")),
            },
            Expr::Filter(v, n, args) => {
                self.expr(v)?;
                let i = self.call_args(n, args)?;
                self.emit(I::Filter(i));
            }
            Expr::Test(v, n, args) => {
                self.expr(v)?;
                let i = self.call_args(n, args)?;
                self.emit(I::Test(i));
            }
        }
        Ok(())
    }
    fn expression(&mut self, s: &str) -> Result<()> {
        let mut p = Parser::new(s)?;
        let e = p.expression(0)?;
        p.finish()?;
        self.expr(&e)
    }
    fn body(&mut self, ends: &[&str]) -> Result<Option<String>> {
        self.depth += 1;
        if self.depth > 128 {
            return Err(Error::new("template nesting limit exceeded"));
        }
        while self.at < self.parts.len() {
            let part = &self.parts[self.at];
            self.offset = part.span.start;
            match &part.kind {
                PartKind::Text(s) => {
                    let v = self.constant(s.clone().into());
                    self.emit(Instruction::Text(v));
                    self.at += 1;
                }
                PartKind::Expression(s) => {
                    let s = s.clone();
                    self.at += 1;
                    self.expression(&s)?;
                    if let Some(Instruction::Join(count, strict)) =
                        self.template.code.last().copied()
                    {
                        *self.template.code.last_mut().unwrap() =
                            Instruction::EmitJoined(count, strict);
                    } else {
                        self.emit(Instruction::Emit);
                    }
                }
                PartKind::Statement(s) => {
                    let s = s.clone();
                    let span = part.span.clone();
                    self.at += 1;
                    let (keyword, rest) = s.split_once(char::is_whitespace).unwrap_or((&s, ""));
                    if ends.contains(&keyword) {
                        self.depth -= 1;
                        return Ok(Some(s));
                    }
                    match keyword {
                        "if" => self.if_block(rest)?,
                        "for" => self.for_block(rest)?,
                        "set" => self.set(rest)?,
                        "macro" => self.macro_block(rest)?,
                        "break" | "continue" => {
                            let id = *self
                                .loops
                                .last()
                                .ok_or_else(|| Error::new("loop control outside a loop"))?;
                            self.emit(if keyword == "break" {
                                Instruction::Break(id)
                            } else {
                                Instruction::Continue(id)
                            });
                        }
                        "mark" => {
                            if self.captures != 0 {
                                return Err(Error::new(
                                    "mark is only valid in root output, outside macros and captures",
                                ));
                            }
                            self.template.annotations.push(span);
                            if rest.is_empty() {
                                self.emit(Instruction::Mark);
                            } else {
                                self.expression(rest)?;
                                self.emit(Instruction::MarkPosition);
                            }
                        }
                        "do" => {
                            self.expression(rest)?;
                            self.emit(Instruction::Drop);
                        }
                        "filter" => {
                            self.emit(Instruction::Capture);
                            self.captures += 1;
                            self.body(&["endfilter"])?;
                            self.captures -= 1;
                            self.emit(Instruction::EndCapture);
                            let mut p = Parser::new(rest)?;
                            let name = p.name()?;
                            let args = if p.eat("(") {
                                p.args()?
                            } else {
                                Args::default()
                            };
                            p.finish()?;
                            let i = self.call_args(&name, &args)?;
                            self.emit(Instruction::Filter(i));
                            self.emit(Instruction::Emit);
                        }
                        _ => return Err(Error::new(format!("unsupported statement {keyword}"))),
                    }
                }
            }
        }
        self.depth -= 1;
        if !ends.is_empty() {
            return Err(Error::new(format!("expected {}", ends.join(" or "))));
        }
        Ok(None)
    }
    fn if_block(&mut self, s: &str) -> Result<()> {
        let mut condition = s.to_owned();
        let mut exits = Vec::new();
        loop {
            self.expression(&condition)?;
            let branch = self.emit(Instruction::JumpFalse(0));
            let stop = self.body(&["elif", "else", "endif"])?.unwrap();
            exits.push(self.emit(Instruction::Jump(0)));
            self.jump(branch, self.pc());
            if let Some(s) = stop.strip_prefix("elif ") {
                condition = s.into();
                continue;
            }
            if stop == "else" {
                self.body(&["endif"])?;
            } else if stop != "endif" {
                return Err(Error::new("invalid conditional terminator"));
            }
            break;
        }
        for i in exits {
            self.jump(i, self.pc());
        }
        Ok(())
    }
    fn for_block(&mut self, s: &str) -> Result<()> {
        let mut p = Parser::new(s)?;
        let mut variables = Vec::new();
        loop {
            variables.push(self.symbol(&p.name()?));
            if !p.eat(",") {
                break;
            }
        }
        p.require("in")?;
        let e = p.expression(1)?;
        let filter = if p.eat("if") {
            Some(p.expression(0)?)
        } else {
            None
        };
        p.finish()?;
        self.expr(&e)?;
        let filter = if let Some(filter) = filter {
            let skip = self.emit(Instruction::Jump(0));
            let start = self.pc();
            self.expr(&filter)?;
            self.emit(Instruction::Return);
            self.jump(skip, self.pc());
            Some(start)
        } else {
            None
        };
        let id = self.template.loops.len() as u32;
        self.template.loops.push(Loop {
            variables,
            filter,
            body: 0,
            empty: 0,
            end: 0,
            next: 0,
        });
        self.emit(Instruction::For(id));
        self.template.loops[id as usize].body = self.pc();
        self.loops.push(id);
        let stop = self.body(&["else", "endfor"])?.unwrap();
        self.loops.pop();
        self.template.loops[id as usize].next = self.pc();
        self.emit(Instruction::Next(id));
        self.template.loops[id as usize].empty = self.pc();
        if stop == "else" {
            self.body(&["endfor"])?;
        }
        self.template.loops[id as usize].end = self.pc();
        self.emit(Instruction::EndLoop);
        Ok(())
    }
    fn set(&mut self, s: &str) -> Result<()> {
        let mut p = Parser::new(s)?;
        let name = p.name()?;
        let slot = self.symbol(&name);
        let attr = if p.eat(".") { Some(p.name()?) } else { None };
        if p.eat("=") {
            let e = p.expression(0)?;
            p.finish()?;
            if let Some(attr) = &attr {
                if let Expr::Binary(Binary::Add, left, right) = &e {
                    if matches!(left.as_ref(),Expr::Attr(v,a) if a==attr && matches!(v.as_ref(),Expr::Name(n) if n==&name))
                    {
                        self.expr(right)?;
                        let attr = self.symbol(attr);
                        self.emit(Instruction::AppendAttr(slot, attr));
                        return Ok(());
                    }
                }
            }
            self.expr(&e)?;
        } else {
            p.finish()?;
            self.emit(Instruction::Capture);
            self.captures += 1;
            self.body(&["endset"])?;
            self.captures -= 1;
            self.emit(Instruction::EndCapture);
        }
        if let Some(attr) = attr {
            let attr = self.symbol(&attr);
            self.emit(Instruction::StoreAttr(slot, attr));
        } else {
            self.emit(Instruction::Store(slot));
        }
        Ok(())
    }
    fn macro_block(&mut self, s: &str) -> Result<()> {
        let mut p = Parser::new(s)?;
        let name = p.name()?;
        let symbol = self.symbol(&name);
        p.require("(")?;
        let mut parameters = Vec::new();
        let mut defaults = Vec::new();
        if !p.eat(")") {
            loop {
                let name = p.name()?;
                let slot = self.symbol(&name);
                parameters.push(slot);
                if p.eat("=") {
                    defaults.push((slot, p.expression(0)?));
                }
                if p.eat(")") {
                    break;
                }
                p.require(",")?;
                if p.eat(")") {
                    break;
                }
            }
        }
        p.finish()?;
        let id = self.template.macros.len() as u32;
        self.template.macros.push(Macro {
            parameters,
            entry: 0,
            defaults: Vec::new(),
            captures: Vec::new(),
        });
        self.emit(Instruction::Macro(symbol, id));
        let skip = self.emit(Instruction::Jump(0));
        for (slot, e) in defaults {
            let pc = self.pc();
            self.expr(&e)?;
            self.emit(Instruction::Return);
            self.template.macros[id as usize].defaults.push((slot, pc));
        }
        let entry = self.pc();
        self.template.macros[id as usize].entry = entry;
        self.captures += 1;
        let loops = std::mem::take(&mut self.loops);
        self.body(&["endmacro"])?;
        self.loops = loops;
        self.captures -= 1;
        self.emit(Instruction::Return);
        let mut captures = Vec::new();
        for op in &self.template.code[skip + 1..] {
            let symbol = match op {
                Instruction::Load(s)
                | Instruction::StoreAttr(s, _)
                | Instruction::AppendAttr(s, _) => Some(*s),
                Instruction::Call(i) => Some(self.template.calls[*i as usize].name),
                _ => None,
            };
            if let Some(symbol) = symbol {
                if !captures.contains(&symbol)
                    && !self.template.macros[id as usize]
                        .parameters
                        .contains(&symbol)
                {
                    captures.push(symbol);
                }
            }
        }
        self.template.macros[id as usize].captures = captures;
        self.jump(skip, self.pc());
        Ok(())
    }
}
