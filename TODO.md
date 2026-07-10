# TODO

## Challenge submission

- Run the container as a Nebius Serverless AI Job and record its job ID, final status, timestamps, image digest, runtime, and billed cost.
- Add redacted Serverless AI Job logs or screenshots as proof of execution.
- Improve the model or recalibrate the release gate; the focused LoRA run reached `0.6875` expected-content accuracy against the current `0.9500` threshold.
- Publish the reviewed adapter, evaluation reports, and GGUF to a public Hugging Face model repository with immutable revisions and recorded hashes.
- Replace all remaining registry, model-repository, secret-selector, and subnet placeholders with exact reproducible instructions after the Serverless AI run.
- Write and publish the required 600-word technical post with `#NebiusServerlessChallenge` and links to the public repositories.
- Record a 3–10 minute public walkthrough video.

## Project polish

- Add a compact architecture diagram and a base-versus-LoRA results visualization.
- Verify a fresh-clone run of every documented command in a clean environment.
