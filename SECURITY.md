# Release hygiene

Keep credentials in your environment or your provider's local credential store,
never in source code, configuration examples, notebooks, or command history.
Generated experiment keys, token sequences, datasets, checkpoints, and outputs
are excluded from Git. Example seeds and sentences are public demonstration
values: generate fresh private keys for any real ownership-verification use.

Only load checkpoints and serialized datasets that you created or trust. Some
vision experiment files contain Python dataset objects and require pickle-based
loading. Do not load untrusted `.pth` files.

The release was created from an allowlisted file snapshot with a new Git history;
no research-repository history is included. A clean scan is not a guarantee that
all possible sensitive information has been detected. If a credential is ever
committed, revoke it rather than relying on deletion alone.
