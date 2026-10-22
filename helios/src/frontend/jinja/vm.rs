use std::cell::{Cell, RefCell};
use std::rc::Rc;

use serde_json::{Map, Value as Json};

use super::compiler::{Binary, Instruction as I, Unary};
use super::value::{LoopValue, Value};
use super::{CompiledTemplate, Error, Mark, RenderedTemplate, Result};

struct Scope {
    undo: usize,
    id: usize,
    closures: usize,
}
struct Closure<'a> {
    definition: u32,
    scope: usize,
    captured: Option<Vec<Value<'a>>>,
}
struct Vm<'a> {
    template: &'a CompiledTemplate,
    slots: Vec<Value<'a>>,
    stamps: Vec<usize>,
    scope: usize,
    serial: usize,
    scopes: Vec<Scope>,
    closures: Vec<Closure<'a>>,
    undo: Vec<(u32, Value<'a>, usize)>,
    stack: Vec<Value<'a>>,
    loops: Vec<Rc<LoopValue<'a>>>,
    output: String,
    captures: Vec<(String, Vec<usize>)>,
    hint_positions: Vec<usize>,
    marks: Vec<Mark>,
    depth: usize,
    hint_texts: Vec<&'a str>,
}

pub(super) fn render(
    template: &CompiledTemplate,
    context: &Map<String, Json>,
    hints: &[(usize, usize)],
) -> Result<RenderedTemplate> {
    let mut hint_texts = Vec::new();
    for &(message, count) in hints {
        let Some(message) = context
            .get("messages")
            .and_then(|messages| messages.get(message))
        else {
            continue;
        };
        let Some(content) = message.get("content") else {
            continue;
        };
        let text = match content {
            Json::String(text) if count == 1 => Some(text.as_str()),
            Json::Array(parts) => count
                .checked_sub(1)
                .and_then(|index| parts.get(index))
                .and_then(|part| part.get("text"))
                .and_then(Json::as_str),
            _ => None,
        };
        if let Some(text) = text {
            hint_texts.push(text);
        }
    }
    let mut vm = Vm {
        template,
        slots: template
            .names
            .iter()
            .map(|s| context.get(s).map(Value::json).unwrap_or_default())
            .collect(),
        stamps: vec![0; template.names.len()],
        scope: 0,
        serial: 0,
        scopes: Vec::new(),
        closures: Vec::new(),
        undo: Vec::new(),
        stack: Vec::with_capacity(16),
        loops: Vec::new(),
        output: String::new(),
        captures: Vec::new(),
        hint_positions: Vec::new(),
        marks: Vec::new(),
        depth: 0,
        hint_texts,
    };
    vm.run(0)?;
    vm.marks.extend(vm.hint_positions.into_iter().map(Mark::Hint));
    Ok(RenderedTemplate {
        text: vm.output,
        marks: vm.marks,
    })
}

impl<'a> Vm<'a> {
    fn pop(&mut self) -> Value<'a> {
        self.stack
            .pop()
            .expect("compiler balances the operand stack")
    }
    fn enter(&mut self) {
        self.scopes.push(Scope {
            undo: self.undo.len(),
            id: self.scope,
            closures: self.closures.len(),
        });
        self.serial += 1;
        self.scope = self.serial;
    }
    fn leave(&mut self) {
        for i in self.scopes.last().unwrap().closures..self.closures.len() {
            let closure = &self.closures[i];
            if closure.scope == self.scope && closure.captured.is_none() {
                let values = self.template.macros[closure.definition as usize]
                    .captures
                    .iter()
                    .map(|&s| self.slots[s as usize].clone())
                    .collect();
                self.closures[i].captured = Some(values);
            }
        }
        let scope = self.scopes.pop().unwrap();
        for (slot, value, stamp) in self.undo.drain(scope.undo..).rev() {
            self.slots[slot as usize] = value;
            self.stamps[slot as usize] = stamp;
        }
        self.scope = scope.id;
    }
    fn lexical(&self, slot: u32, scope: usize) -> Value<'a> {
        let mut value = &self.slots[slot as usize];
        let mut stamp = self.stamps[slot as usize];
        if stamp > scope {
            for (s, previous, previous_stamp) in self.undo.iter().rev() {
                if *s == slot {
                    value = previous;
                    stamp = *previous_stamp;
                    if stamp <= scope {
                        break;
                    }
                }
            }
        }
        value.clone()
    }
    fn store(&mut self, slot: u32, value: Value<'a>) {
        let value = self.hinted(value);
        let index = slot as usize;
        if self.scope != 0 && self.stamps[index] != self.scope {
            self.undo.push((
                slot,
                std::mem::take(&mut self.slots[index]),
                self.stamps[index],
            ));
            self.stamps[index] = self.scope;
        }
        self.slots[index] = value;
    }
    fn iteration(&mut self, id: u32) -> Result<()> {
        let value = self.loops.last().unwrap().clone();
        let item = value.items[value.index.get()].clone();
        self.store(self.template.loop_symbol, Value::Loop(value));
        self.loop_variables(id, item)
    }
    fn loop_variables(&mut self, id: u32, item: Value<'a>) -> Result<()> {
        let vars = &self.template.loops[id as usize].variables;
        if vars.len() == 1 {
            self.store(vars[0], item);
        } else {
            let items = item.iter()?;
            if items.len() != vars.len() {
                return Err(Error::new("cannot unpack sequence of different length"));
            }
            for (&slot, value) in vars.iter().zip(items) {
                self.store(slot, value);
            }
        }
        Ok(())
    }
    fn hinted(&self, value: Value<'a>) -> Value<'a> {
        if let Value::Str(text) = value
            && self
                .hint_texts
                .iter()
                .any(|hint| !text.is_empty() && std::ptr::eq(*hint, text))
        {
            return Value::marked(text.to_owned(), vec![text.len()]);
        }
        value
    }
    fn emit(&mut self, value: Value<'a>) {
        self.hinted(value)
            .append(&mut self.output, &mut self.hint_positions);
    }
    fn capture(&mut self) {
        self.captures.push((
            std::mem::take(&mut self.output),
            std::mem::take(&mut self.hint_positions),
        ));
    }
    fn end_capture(&mut self) -> Value<'a> {
        let (output, hints) = self.captures.pop().unwrap();
        let text = std::mem::replace(&mut self.output, output);
        let positions = std::mem::replace(&mut self.hint_positions, hints);
        Value::marked(text, positions)
    }
    fn run(&mut self, start: u32) -> Result<()> {
        self.depth += 1;
        if self.depth > 128 {
            return Err(Error::new("macro recursion limit exceeded"));
        }
        let mut pc = start as usize;
        let result = (|| {
            while pc < self.template.code.len() {
                let op = self.template.code[pc];
                pc += 1;
                match op {
                    I::Text(id) => self
                        .output
                        .push_str(self.template.constants[id as usize].as_str().unwrap()),
                    I::Constant(id) => self
                        .stack
                        .push(Value::json(&self.template.constants[id as usize])),
                    I::Load(id) => self.stack.push(self.slots[id as usize].clone()),
                    I::Store(id) => {
                        let v = self.pop();
                        self.store(id, v);
                    }
                    I::StoreAttr(slot, name) | I::AppendAttr(slot, name) => {
                        let value = self.pop();
                        let Value::Namespace(ns) = &self.slots[slot as usize] else {
                            return Err(Error::new(
                                "cannot assign attribute of non-namespace object",
                            ));
                        };
                        let mut ns = ns.borrow_mut();
                        let name = &self.template.names[name as usize];
                        if let Some((_, old)) = ns.iter_mut().find(|(key, _)| key == name) {
                            if matches!(op, I::AppendAttr(..)) {
                                match (old, value) {
                                    (Value::List(items), value)
                                        if value.sequence() && value.as_str().is_none() =>
                                    {
                                        Rc::make_mut(items).extend(value.iter()?)
                                    }
                                    (old, value) => {
                                        *old = binary(Binary::Add, std::mem::take(old), value)?
                                    }
                                }
                            } else {
                                *old = value;
                            }
                        } else if matches!(op, I::AppendAttr(..)) {
                            return Err(Error::new("undefined namespace attribute"));
                        } else {
                            ns.push((name.clone(), value));
                        }
                    }
                    I::Attr(name) => {
                        let v = self.pop();
                        if matches!(v, Value::Undefined) {
                            return Err(Error::new("undefined value"));
                        }
                        let value = self.hinted(v.attr(&self.template.names[name as usize]));
                        self.stack.push(value);
                    }
                    I::Index => {
                        let key = self.pop();
                        let v = self.pop();
                        if matches!(v, Value::Undefined) {
                            return Err(Error::new("undefined value"));
                        }
                        let value = self.hinted(v.index(&key));
                        self.stack.push(value);
                    }
                    I::Slice => {
                        let step = self.pop();
                        let end = self.pop();
                        let begin = self.pop();
                        let v = self.pop();
                        self.stack.push(slice(v, begin, end, step)?);
                    }
                    I::List(n) => {
                        let items = self.stack.split_off(self.stack.len() - n as usize);
                        self.stack.push(Value::list(items));
                    }
                    I::Map(n) => {
                        let values = self.stack.split_off(self.stack.len() - 2 * n as usize);
                        let mut pairs = Vec::with_capacity(n as usize);
                        let mut values = values.into_iter();
                        while let Some(key) = values.next() {
                            pairs.push((key.text(), values.next().unwrap()));
                        }
                        self.stack.push(Value::Map(Rc::new(pairs)));
                    }
                    I::Unary(op) => {
                        let v = self.pop();
                        self.stack.push(match op {
                            Unary::Not => Value::Bool(!v.truthy()),
                            Unary::Pos => v,
                            Unary::Neg => match v {
                                Value::Int(v) => Value::Int(
                                    v.checked_neg()
                                        .ok_or_else(|| Error::new("integer overflow"))?,
                                ),
                                _ => Value::Float(
                                    -v.number()
                                        .ok_or_else(|| Error::new("cannot negate non-number"))?,
                                ),
                            },
                        });
                    }
                    I::Binary(op) => {
                        let b = self.pop();
                        let a = self.pop();
                        self.stack.push(binary(op, a, b)?);
                    }
                    I::Test(id) | I::Filter(id) | I::Method(id) | I::Call(id) => {
                        let call = &self.template.calls[id as usize];
                        let count = call.positional + call.keywords.len();
                        let arguments = self.stack.split_off(self.stack.len() - count);
                        let name = self.template.names[call.name as usize].as_str();
                        let (args, kw) = arguments.split_at(call.positional);
                        let kwargs: Vec<_> = call
                            .keywords
                            .iter()
                            .zip(kw)
                            .map(|(key, v)| {
                                (self.template.names[*key as usize].as_str(), v.clone())
                            })
                            .collect();
                        let result = match op {
                            I::Call(_) => {
                                if let Value::Macro(id) = self.slots[call.name as usize] {
                                    let direct =
                                        matches!(self.template.code.get(pc), Some(I::Emit));
                                    let value = self.macro_call(id, args, &kwargs, direct)?;
                                    if direct {
                                        pc += 1;
                                        continue;
                                    }
                                    value
                                } else {
                                    function(name, args, &kwargs)?
                                }
                            }
                            I::Test(_) => {
                                let v = self.pop();
                                Value::Bool(test(name, &v, args)?)
                            }
                            I::Filter(_) => {
                                let v = self.pop();
                                filter(name, v, args, &kwargs)?
                            }
                            _ => {
                                let v = self.pop();
                                method(name, v, args, &kwargs)?
                            }
                        };
                        self.stack.push(result);
                    }
                    I::Emit => {
                        let v = self.pop();
                        self.emit(v);
                    }
                    I::Join(count, strict) | I::EmitJoined(count, strict) => {
                        let start = self.stack.len() - count as usize;
                        let values = &self.stack[start..];
                        if strict && values.iter().any(|v| v.as_str().is_none()) {
                            return Err(Error::new("string addition requires strings"));
                        }
                        let size = values.iter().map(|v| v.as_str().map_or(0, str::len)).sum();
                        if matches!(op, I::EmitJoined(..)) {
                            self.output.reserve(size);
                            for index in start..self.stack.len() {
                                self.emit(self.stack[index].clone());
                            }
                            self.stack.truncate(start);
                        } else {
                            let mut text = String::with_capacity(size);
                            let mut hints = Vec::new();
                            for v in self.stack.drain(start..) {
                                v.append(&mut text, &mut hints);
                            }
                            self.stack.push(Value::marked(text, hints));
                        }
                    }
                    I::Drop => {
                        self.pop();
                    }
                    I::Jump(target) => pc = target as usize,
                    I::JumpFalse(target) => {
                        if !self.pop().truthy() {
                            pc = target as usize;
                        }
                    }
                    I::And(target) => {
                        if !self.stack.last().unwrap().truthy() {
                            pc = target as usize;
                        } else {
                            self.pop();
                        }
                    }
                    I::Or(target) => {
                        if self.stack.last().unwrap().truthy() {
                            pc = target as usize;
                        } else {
                            self.pop();
                        }
                    }
                    I::For(id) => {
                        let mut items = self.pop().iter()?;
                        if let Some(filter) = self.template.loops[id as usize].filter {
                            self.enter();
                            let mut selected = Vec::new();
                            for item in items {
                                self.loop_variables(id, item.clone())?;
                                self.run(filter)?;
                                if self.pop().truthy() {
                                    selected.push(item);
                                }
                            }
                            self.leave();
                            items = selected;
                        }
                        let empty = items.is_empty();
                        self.loops.push(Rc::new(LoopValue {
                            items,
                            index: Cell::new(0),
                        }));
                        self.enter();
                        if empty {
                            pc = self.template.loops[id as usize].empty as usize;
                        } else {
                            self.iteration(id)?;
                        }
                    }
                    I::Next(id) => {
                        let lp = self.loops.last().unwrap();
                        let next = lp.index.get() + 1;
                        lp.index.set(next);
                        if next < lp.items.len() {
                            self.leave();
                            self.enter();
                            self.iteration(id)?;
                            pc = self.template.loops[id as usize].body as usize;
                        } else {
                            pc = self.template.loops[id as usize].end as usize;
                        }
                    }
                    I::EndLoop => {
                        self.loops.pop();
                        self.leave();
                    }
                    I::Break(id) => pc = self.template.loops[id as usize].end as usize,
                    I::Continue(id) => pc = self.template.loops[id as usize].next as usize,
                    I::Macro(slot, id) => {
                        let closure = self.closures.len() as u32;
                        self.closures.push(Closure {
                            definition: id,
                            scope: self.scope,
                            captured: None,
                        });
                        self.store(slot, Value::Macro(closure));
                    }
                    I::Return => break,
                    I::Capture => self.capture(),
                    I::EndCapture => {
                        let value = self.end_capture();
                        self.stack.push(value);
                    }
                    I::Mark => self.marks.push(Mark::Byte(self.output.len())),
                    I::MarkPosition => {
                        let Value::Int(v) = self.pop() else {
                            return Err(Error::new("mark position must be a nonnegative integer"));
                        };
                        let v = usize::try_from(v).map_err(|_| {
                            Error::new("mark position must be a nonnegative integer")
                        })?;
                        self.marks.push(Mark::Position(v));
                    }
                }
            }
            Ok(())
        })();
        self.depth -= 1;
        result.map_err(|mut error: Error| {
            if error.offset == 0 {
                error.offset = self.template.spans[pc.saturating_sub(1)];
            }
            error
        })
    }
    fn macro_call(
        &mut self,
        id: u32,
        args: &[Value<'a>],
        kwargs: &[(&str, Value<'a>)],
        direct: bool,
    ) -> Result<Value<'a>> {
        let closure = &self.closures[id as usize];
        let scope = closure.scope;
        let mac = &self.template.macros[closure.definition as usize];
        if args.len() > mac.parameters.len() {
            return Err(Error::new("too many arguments to macro"));
        }
        self.enter();
        for (i, &slot) in mac.captures.iter().enumerate() {
            let value = if let Some(captured) = &self.closures[id as usize].captured {
                captured[i].clone()
            } else {
                self.lexical(slot, scope)
            };
            self.store(slot, value);
        }
        for (i, &slot) in mac.parameters.iter().enumerate() {
            let name = self.template.names[slot as usize].as_str();
            let value = if let Some(value) = args.get(i) {
                value.clone()
            } else if let Some((_, value)) = kwargs.iter().find(|(k, _)| *k == name) {
                value.clone()
            } else if let Some((_, pc)) = mac.defaults.iter().find(|(s, _)| *s == slot) {
                self.run(*pc)?;
                self.pop()
            } else {
                Value::Undefined
            };
            self.store(slot, value);
        }
        for (name, _) in kwargs {
            if !mac
                .parameters
                .iter()
                .any(|&s| self.template.names[s as usize] == *name)
            {
                return Err(Error::new(format!("unknown macro argument {name}")));
            }
        }
        if !direct {
            self.capture();
        }
        self.run(mac.entry)?;
        self.leave();
        Ok(if !direct {
            self.end_capture()
        } else {
            Value::Undefined
        })
    }
}

fn binary<'a>(op: Binary, a: Value<'a>, b: Value<'a>) -> Result<Value<'a>> {
    use Binary as B;
    match op {
        B::Eq => return Ok(Value::Bool(a.equals(&b))),
        B::Ne => return Ok(Value::Bool(!a.equals(&b))),
        B::In | B::NotIn => {
            let found = if let Some(text) = b.as_str() {
                text.contains(
                    a.as_str()
                        .ok_or_else(|| Error::new("string membership requires a string"))?,
                )
            } else if b.mapping() {
                a.as_str()
                    .is_some_and(|key| !matches!(b.attr(key), Value::Undefined))
            } else {
                b.iter()?.iter().any(|v| v.equals(&a))
            };
            return Ok(Value::Bool(found != matches!(op, B::NotIn)));
        }
        B::Concat => {
            let mut text = String::new();
            let mut hints = Vec::new();
            a.append(&mut text, &mut hints);
            b.append(&mut text, &mut hints);
            return Ok(Value::marked(text, hints));
        }
        B::Add if a.as_str().is_some() && b.as_str().is_some() => {
            let mut text = String::new();
            let mut hints = Vec::new();
            a.append(&mut text, &mut hints);
            b.append(&mut text, &mut hints);
            return Ok(Value::marked(text, hints));
        }
        B::Add if a.sequence() && b.sequence() && a.as_str().is_none() && b.as_str().is_none() => {
            let mut items = a.iter()?;
            items.extend(b.iter()?);
            return Ok(Value::list(items));
        }
        B::Mul if a.as_str().is_some() && b.integer().is_some() => {
            return Ok(Value::string(
                a.as_str()
                    .unwrap()
                    .repeat(b.integer().unwrap().max(0) as usize),
            ));
        }
        B::Lt | B::Le | B::Gt | B::Ge => {
            let ordering = if let (Some(a), Some(b)) = (a.as_str(), b.as_str()) {
                Some(a.cmp(b))
            } else if let (Some(a), Some(b)) = (a.number(), b.number()) {
                a.partial_cmp(&b)
            } else {
                None
            };
            let ordering = ordering.ok_or_else(|| Error::new("cannot compare values"))?;
            return Ok(Value::Bool(match op {
                B::Lt => ordering.is_lt(),
                B::Le => ordering.is_le(),
                B::Gt => ordering.is_gt(),
                _ => ordering.is_ge(),
            }));
        }
        _ => {}
    }
    if let (Some(a), Some(b)) = (a.integer(), b.integer()) {
        let value = match op {
            B::Add => a.checked_add(b),
            B::Sub => a.checked_sub(b),
            B::Mul => a.checked_mul(b),
            B::FloorDiv if b != 0 => a.checked_div_euclid(b),
            B::Rem if b != 0 => a.checked_rem_euclid(b),
            B::Pow if b >= 0 => u32::try_from(b).ok().and_then(|b| a.checked_pow(b)),
            _ => None,
        };
        if let Some(value) = value {
            return Ok(Value::Int(value));
        }
    }
    let a = a
        .number()
        .ok_or_else(|| Error::new("arithmetic requires compatible numbers or sequences"))?;
    let b = b
        .number()
        .ok_or_else(|| Error::new("arithmetic requires compatible numbers or sequences"))?;
    if b == 0.0 && matches!(op, B::Div | B::FloorDiv | B::Rem) {
        return Err(Error::new("division by zero"));
    }
    Ok(Value::Float(match op {
        B::Add => a + b,
        B::Sub => a - b,
        B::Mul => a * b,
        B::Div => a / b,
        B::FloorDiv => (a / b).floor(),
        B::Rem => a - b * (a / b).floor(),
        B::Pow => a.powf(b),
        _ => unreachable!(),
    }))
}

fn slice<'a>(
    value: Value<'a>,
    begin: Value<'a>,
    end: Value<'a>,
    step: Value<'a>,
) -> Result<Value<'a>> {
    let n = value.len() as i64;
    let integer = |v: &Value<'_>, default| match v {
        Value::None => Ok(default),
        Value::Int(v) => Ok(*v),
        _ => Err(Error::new("slice indices must be integers")),
    };
    let step = integer(&step, 1)?;
    if step == 0 {
        return Err(Error::new("slice step cannot be zero"));
    }
    let bound = |v: &Value<'_>, default| -> Result<i64> {
        if matches!(v, Value::None) {
            return Ok(default);
        }
        let mut v = integer(v, default)?;
        if v < 0 {
            v += n;
        }
        Ok(if step < 0 {
            v.clamp(-1, n - 1)
        } else {
            v.clamp(0, n)
        })
    };
    let mut i = bound(&begin, if step < 0 { n - 1 } else { 0 })?;
    let end = bound(&end, if step < 0 { -1 } else { n })?;
    let mut items = Vec::new();
    while if step < 0 { i > end } else { i < end } {
        items.push(value.index(&Value::Int(i)));
        i = i
            .checked_add(step)
            .ok_or_else(|| Error::new("slice index overflow"))?;
    }
    if value.as_str().is_some() {
        let mut s = String::new();
        for v in items {
            v.write(&mut s);
        }
        Ok(Value::string(s))
    } else {
        Ok(Value::list(items))
    }
}

fn function<'a>(name: &str, args: &[Value<'a>], kw: &[(&str, Value<'a>)]) -> Result<Value<'a>> {
    match name {
        "namespace" => {
            let mut pairs = if let Some(v) = args.first() {
                v.pairs()?
            } else {
                Vec::new()
            };
            pairs.extend(kw.iter().map(|(k, v)| ((*k).into(), v.clone())));
            Ok(Value::Namespace(Rc::new(RefCell::new(pairs))))
        }
        "range" => {
            if args.is_empty() || args.len() > 3 {
                return Err(Error::new("range expects one to three arguments"));
            }
            let ints = args
                .iter()
                .map(|v| {
                    v.integer()
                        .ok_or_else(|| Error::new("range expects integers"))
                })
                .collect::<Result<Vec<_>>>()?;
            let (mut i, end, step) = match ints.as_slice() {
                [end] => (0, *end, 1),
                [start, end] => (*start, *end, 1),
                [start, end, step] => (*start, *end, *step),
                _ => unreachable!(),
            };
            if step == 0 {
                return Err(Error::new("range step cannot be zero"));
            }
            let mut items = Vec::new();
            while if step < 0 { i > end } else { i < end } {
                items.push(Value::Int(i));
                i = i
                    .checked_add(step)
                    .ok_or_else(|| Error::new("range overflow"))?;
            }
            Ok(Value::list(items))
        }
        "raise_exception" => Err(Error::new(
            args.first().map(Value::text).unwrap_or_default(),
        )),
        "dict" => Ok(Value::Map(Rc::new(
            kw.iter().map(|(k, v)| ((*k).into(), v.clone())).collect(),
        ))),
        _ => Err(Error::new(format!("unknown function {name}"))),
    }
}

fn test<'a>(name: &str, v: &Value<'a>, args: &[Value<'a>]) -> Result<bool> {
    Ok(match name {
        "defined" => !matches!(v, Value::Undefined),
        "undefined" => matches!(v, Value::Undefined),
        "none" => matches!(v, Value::None),
        "true" => matches!(v, Value::Bool(true)),
        "false" => matches!(v, Value::Bool(false)),
        "boolean" => matches!(v, Value::Bool(_)),
        "string" => v.as_str().is_some(),
        "mapping" => v.mapping(),
        "sequence" => v.sequence() || v.mapping(),
        "iterable" => v.iterable(),
        "number" => v.number().is_some(),
        "integer" => matches!(v, Value::Int(_) | Value::UInt(_)),
        "float" => matches!(v, Value::Float(_)),
        "equalto" | "eq" => args.first().is_some_and(|a| v.equals(a)),
        "odd" => v.integer().is_some_and(|v| v % 2 != 0),
        "even" => v.integer().is_some_and(|v| v % 2 == 0),
        _ => return Err(Error::new(format!("unknown test {name}"))),
    })
}

fn filter<'a>(
    name: &str,
    v: Value<'a>,
    args: &[Value<'a>],
    kw: &[(&str, Value<'a>)],
) -> Result<Value<'a>> {
    match name {
        "safe" => Ok(v),
        "default" | "d" => Ok(
            if matches!(v, Value::Undefined)
                || args.get(1).is_some_and(Value::truthy) && !v.truthy()
            {
                args.first().cloned().unwrap_or(Value::Str(""))
            } else {
                v
            },
        ),
        "length" | "count" => Ok(Value::Int(v.len() as i64)),
        "string" => Ok(if v.as_str().is_some() {
            v
        } else {
            Value::string(v.text())
        }),
        "trim" => {
            let chars = args.first().and_then(Value::as_str);
            let pred = |c: char| chars.map_or_else(|| c.is_whitespace(), |s| s.contains(c));
            if let Value::Str(s) = v {
                return Ok(Value::Str(s.trim_matches(pred)));
            }
            let text = if let Some(s) = v.as_str() {
                s
            } else {
                return Ok(Value::string(v.text().trim_matches(pred).into()));
            };
            let trimmed = text.trim_matches(pred);
            if trimmed.len() == text.len() {
                Ok(v)
            } else {
                let start = trimmed.as_ptr() as usize - text.as_ptr() as usize;
                let hints = v
                    .hints()
                    .iter()
                    .filter(|end| **end > start && !trimmed.is_empty())
                    .map(|end| (end - start).min(trimmed.len()))
                    .collect();
                Ok(Value::marked(trimmed.into(), hints))
            }
        }
        "capitalize" => {
            let text = v.text().to_lowercase();
            let mut chars = text.chars();
            Ok(Value::string(
                chars
                    .next()
                    .map(|c| c.to_uppercase().to_string() + chars.as_str())
                    .unwrap_or_default(),
            ))
        }
        "lower" => Ok(Value::string(v.text().to_lowercase())),
        "upper" => Ok(Value::string(v.text().to_uppercase())),
        "replace" => {
            let from = args
                .first()
                .ok_or_else(|| Error::new("replace requires two arguments"))?
                .text();
            let to = args
                .get(1)
                .ok_or_else(|| Error::new("replace requires two arguments"))?
                .text();
            Ok(Value::string(
                if let Some(count) = args.get(2).and_then(Value::integer) {
                    v.text().replacen(&from, &to, count.max(0) as usize)
                } else {
                    v.text().replace(&from, &to)
                },
            ))
        }
        "items" => Ok(Value::list(
            v.pairs()?
                .into_iter()
                .map(|(k, v)| Value::list(vec![Value::string(k), v]))
                .collect(),
        )),
        "tojson" => {
            let kwargs = kw
                .iter()
                .map(|(k, v)| Ok(((*k).to_owned(), v.to_json()?)))
                .collect::<Result<Map<_, _>>>()?;
            Ok(Value::string(
                super::super::json::format_json(&v.to_json()?, &kwargs).map_err(Error::new)?,
            ))
        }
        "fromjson" => Ok(Value::owned(
            serde_json::from_str(
                v.as_str()
                    .ok_or_else(|| Error::new("fromjson requires a string"))?,
            )
            .map_err(|e| Error::new(e.to_string()))?,
        )),
        "sort" => {
            let mut items = v.iter()?;
            let attr = kw
                .iter()
                .find(|(k, _)| *k == "attribute")
                .and_then(|(_, v)| v.as_str());
            let reverse = kw
                .iter()
                .find(|(k, _)| *k == "reverse")
                .map(|(_, v)| v.truthy())
                .or_else(|| args.first().map(Value::truthy))
                .unwrap_or(false);
            items.sort_by(|a, b| {
                let a = attr.map_or_else(|| a.clone(), |k| a.attr(k));
                let b = attr.map_or_else(|| b.clone(), |k| b.attr(k));
                let order = if let (Some(a), Some(b)) = (a.number(), b.number()) {
                    a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    a.text().to_lowercase().cmp(&b.text().to_lowercase())
                };
                if reverse { order.reverse() } else { order }
            });
            Ok(Value::list(items))
        }
        "join" => {
            let sep = args.first().map(Value::text).unwrap_or_default();
            let mut text = String::new();
            for (i, v) in v.iter()?.iter().enumerate() {
                if i != 0 {
                    text.push_str(&sep);
                }
                v.write(&mut text);
            }
            Ok(Value::string(text))
        }
        "list" => Ok(Value::list(v.iter()?)),
        "first" => Ok(v.index(&Value::Int(0))),
        "last" => Ok(v.index(&Value::Int(-1))),
        "int" => Ok(Value::Int(
            v.integer()
                .or_else(|| v.as_str().and_then(|v| v.parse().ok()))
                .unwrap_or(0),
        )),
        _ => Err(Error::new(format!("unknown filter {name}"))),
    }
}

fn method<'a>(
    name: &str,
    v: Value<'a>,
    args: &[Value<'a>],
    kw: &[(&str, Value<'a>)],
) -> Result<Value<'a>> {
    if v.mapping() {
        return match name {
            "items" => filter("items", v, args, kw),
            "keys" => Ok(Value::list(v.iter()?)),
            "values" => Ok(Value::list(
                v.pairs()?.into_iter().map(|(_, v)| v).collect(),
            )),
            "get" => {
                let key = args
                    .first()
                    .ok_or_else(|| Error::new("get requires a key"))?;
                let value = v.index(key);
                Ok(if matches!(value, Value::Undefined) {
                    args.get(1).cloned().unwrap_or(Value::None)
                } else {
                    value
                })
            }
            _ => Err(Error::new(format!("unknown map method {name}"))),
        };
    }
    let text = v
        .as_str()
        .ok_or_else(|| Error::new(format!("unknown method {name}")))?;
    let argument = |i: usize| {
        args.get(i)
            .and_then(Value::as_str)
            .ok_or_else(|| Error::new(format!("{name} requires string arguments")))
    };
    match name {
        "startswith" => Ok(Value::Bool(text.starts_with(argument(0)?))),
        "endswith" => Ok(Value::Bool(text.ends_with(argument(0)?))),
        "strip" | "lstrip" | "rstrip" => {
            let chars = args.first().and_then(Value::as_str);
            let pred = |c: char| chars.map_or_else(|| c.is_whitespace(), |s| s.contains(c));
            Ok(Value::string(
                match name {
                    "lstrip" => text.trim_start_matches(pred),
                    "rstrip" => text.trim_end_matches(pred),
                    _ => text.trim_matches(pred),
                }
                .into(),
            ))
        }
        "split" | "rsplit" => {
            let limit = args
                .get(1)
                .and_then(Value::integer)
                .filter(|n| *n >= 0)
                .map(|n| n as usize + 1)
                .unwrap_or(usize::MAX);
            let parts: Vec<_> = if args.is_empty() || matches!(args[0], Value::None) {
                text.split_whitespace()
                    .map(|s| Value::string(s.into()))
                    .collect()
            } else {
                let sep = argument(0)?;
                if sep.is_empty() {
                    return Err(Error::new("empty separator"));
                }
                if name == "rsplit" {
                    let mut items: Vec<_> = text
                        .rsplitn(limit, sep)
                        .map(|s| Value::string(s.into()))
                        .collect();
                    items.reverse();
                    items
                } else {
                    text.splitn(limit, sep)
                        .map(|s| Value::string(s.into()))
                        .collect()
                }
            };
            Ok(Value::list(parts))
        }
        "replace" | "lower" | "upper" | "capitalize" => filter(name, v, args, kw),
        _ => Err(Error::new(format!("unknown string method {name}"))),
    }
}
