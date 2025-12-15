# Session Environment & Store

The CAS Engine supports a persistent session state that allows you to store expressions, define variables, and reference previous results. This system mimics the workflow of notebook interfaces (like Mathematica or Jupyter).

## Session Store (History)

Every expression or equation you evaluate is automatically stored in the **Session Store** and assigned a unique **Entry ID** (e.g., `#1`, `#2`).

### Usage
- **Referencing Entries**: You can use `#id` in any subsequent expression to insert the stored value.
  ```text
  > 1 + 2
  3
  #1: 3

  > #1 * 5
  15
  #2: 15
  ```
- **Referencing Equations**: If an entry is an equation (`lhs = rhs`), referencing it as an expression converts it to its **residue form** (`lhs - rhs`).
  ```text
  > x + 1 = 5
  x + 1 = 5
  #1: x + 1 = 5

  > #1
  (x + 1) - 5
  ```

### Commands
- `history` or `list`: Show all stored entries.
- `show #id`: Display a specific entry.
- `del #id ...`: Delete specific entries. Note that IDs are **unique and never reused**, even after deletion.
- `reset`: Clear the entire session history (and environment).

---

## Environment (Variables)

The **Environment** allows you to bind expressions to variable names. These bindings are used for substitution in subsequent calculations.

### Assignment
Use `let` or `:=` to assign variables:
```text
> let a = 10
> b := a + 5
```

### Substitution Logic
Variable substitution is **transitive** and **cycle-safe**:

1. **Transitive**: If `a = b` and `b = 5`, then `a` evaluates to `5`.
2. **Cycle Detection**: Recursive definitions (e.g., `x = x + 1`) are detected and substitution is halted to prevent infinite loops.
3. **Shadowing**: Some commands (like `diff expr, x`) temporarily "shadow" a variable (`x`) to prevent it from being substituted, ensuring it is treated as a symbolic variable even if it has a value in the environment.

### Reserved Names
The following names are reserved and cannot be assigned:
- **Keywords**: `let`, `vars`, `clear`, `solve`, `simplify`, etc.
- **Built-in Functions**: `sin`, `cos`, `log`, `sqrt`, etc.
- **Constants**: `pi`, `e`, `i`, `inf`.

### Commands
- `vars`: List all defined variables and their values.
- `clear <name>`: Remove a specific variable binding.
- `clear`: Remove all variable bindings.
