# Statistics & Sampling — Debugging Practice

You have **60 minutes** to find and fix the bugs in `stats_lib.py`.
The test suite in `test_stats_lib.py` defines the expected behavior.

## Rules

- No AI assistance. No web search for the specific failure messages.
- You may consult: Python stdlib documentation, NumPy documentation.
- Allowed libraries: Python stdlib + NumPy. Nothing else.
- Free tools: `pdb` / `breakpoint()`, `print`, `python3 -c '...'`.

## Run the tests

```bash
python3 -m unittest test_stats_lib -v
```

Run a single test class or method:

```bash
python3 -m unittest test_stats_lib.TestPercentile -v
python3 -m unittest test_stats_lib.TestRunningStats.test_two_instances_do_not_share_state -v
```

## Tips

- Read the failing test first, then the function it exercises.
- Some tests already pass. **Don't break them.** They act as guardrails.
- A few bugs only show up when a function is called more than once, or
  when two instances of the same class coexist. If a test depends on
  prior state, that is a clue about *which* state is leaking.
- `breakpoint()` inserts a `pdb` stop. Use it liberally.
- For numerical issues, print both the value and the type/dtype.

## After the timer

**Do NOT open `BUGS.md` until the timer ends or you've given up debugging.**
That file documents every bug and the expected fix; reading it during
the run defeats the practice.
