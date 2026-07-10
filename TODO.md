# TODO

## Challenge submission

- [x] Run the full container as Nebius Serverless AI Job `aijob-e00kxw7qp6kg6z18w2` and record its immutable inputs, `COMPLETED` status, timestamps, and 37m 24s runtime.
- [x] Add a redacted Serverless AI Job log excerpt and measured results to `docs/experiment-results.md`.
- [ ] Replace the compute-only `$0.97` estimate plus disk with the settled billed cost when the Nebius billing export posts.
- [ ] Improve completion quality without tuning against the current dev suite; the Serverless run reached `0.65625` expected-content accuracy against the `0.9500` release threshold.
- [x] Publish the retained reviewed experimental adapter, evaluation reports, and GGUF adapter to the public Hugging Face model repository with immutable revisions and recorded hashes.
- [x] Record the image digest, public dataset revision, model repository, and credentialless selector used for the proof run; redact the account-scoped registry namespace and subnet.
- [x] Publish the canonical retained 702-row Nebius training package in one Hugging Face root commit and record its immutable revision.
- [x] Publish [Tiny Interjection Model](https://jeremysoojk.substack.com/p/tiny-interjection-model) with `#NebiusServerlessChallenge` and public repository links.
- Record a 3–10 minute public walkthrough video.

## Project polish

- [x] Add README-ready training-loss, base-versus-LoRA, and Serverless progress visualizations.
- [ ] Add a compact architecture diagram.
- Verify a fresh-clone run of every documented command in a clean environment.
