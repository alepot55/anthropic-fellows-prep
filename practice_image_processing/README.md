# Image Processing — Debugging Practice

You have **60 minutes** to find and fix the bugs in `image_lib.py`.
The test suite in `test_image_lib.py` defines the expected behavior.

## Rules

- No AI assistance. No web search for the specific failure messages.
- You may consult: Python stdlib documentation, NumPy documentation.
- Allowed libraries: Python stdlib + NumPy. Nothing else.
- Free tools: `pdb` / `breakpoint()`, `print`, `python3 -c '...'`.

## Run the tests

```bash
python3 -m unittest test_image_lib -v
```

Run a single test:

```bash
python3 -m unittest test_image_lib.TestHistogram.test_partial_range_values -v
```

## Tips

- Read the failing test first, then the function it exercises.
- Some tests already pass. **Don't break them.** They act as guardrails.
- `breakpoint()` inserts a `pdb` stop. Use it liberally.
- If a function returns a wrong shape, `print(result.shape)` is usually
  faster than reasoning about broadcasting in your head.

## After the timer

**Do NOT open `BUGS.md` until the timer ends or you've given up debugging.**
That file documents every bug and the expected fix; reading it during
the run defeats the practice.
