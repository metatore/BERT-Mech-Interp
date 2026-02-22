# Heuristic Lexicons

These files are optional inputs for the cheap heuristic tagger in `src/probes.py`.
If present, they override the built-in defaults.

Format:
- one term per line
- lowercase preferred
- lines starting with `#` are ignored

Recommended bootstrapping strategy:
- `brands.txt`: start from open-source Amazon-like catalog brand lists, then normalize aliases
- `colors.txt`, `spec_units.txt`, `bundle_terms.txt`: taxonomy/attribute vocab sources
- `stopwords.txt`: project-specific lexical noise words

Notes:
- Use token/phrase entries (e.g. `new balance`, `under armour`)
- Keep generic ambiguous words out of `brands.txt` unless they are clearly useful in your domain
