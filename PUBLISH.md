# Publication Checklist

Use this gate before making any code, dataset, model, image, log, or repository public. Do not publish until every applicable item passes.

## Identity and ownership

- [ ] The public name is exactly **Tiny Interjection Model Alpha**; use **TIM** only as its acronym.
- [ ] The repository, package metadata, image tags, new job names, artifact names, API title, documentation, and license use the same current identity. Immutable historical provider records may retain their original names when the documentation identifies them accurately.
- [ ] No legacy product names, acronyms, artifact prefixes, or unrelated organization branding remain anywhere in the tracked tree.
- [ ] `LICENSE` says `Copyright (c) 2026 Tiny Interjection Model Alpha contributors`.
- [ ] The root commit author and committer are the `Jiply` GitHub account using its account-linked private noreply address.
- [ ] GitHub resolves the commit author to the `Jiply` profile; do not use a generic project author or expose a personal email address.

## Privacy and security

- [ ] No unintended personal names or email addresses, home-directory paths, workstation details, account IDs, private repository names, or private URLs appear in tracked files or commit metadata. Intentional maker attribution and approved public repository owners are allowed.
- [ ] No internal CLI wrappers, private setup conventions, local aliases, or machine-specific commands appear in public instructions.
- [ ] No credentials, tokens, API keys, secret selectors, environment dumps, signed URLs, private data, or raw unreviewed cloud logs are committed.
- [ ] No account-scoped cloud resource names, instance IDs, project IDs, subnet IDs, billing details, or other account identifiers are visible in text, screenshots, logs, or image metadata. Retain only the public challenge-proof job ID and immutable image digest.
- [ ] All examples use neutral placeholders such as `ORGANIZATION/DATASET`, `REGISTRY/IMAGE`, `DATASET_COMMIT_SHA`, and `SUBNET_ID`.
- [ ] The exact staged tree—not only the working tree—has been reviewed for PII, secrets, security regressions, privacy regressions, and device-specific paths.

## Required provider provenance

- [ ] Keep the DigitalOcean Serverless Inference teacher-generation code, validation flow, non-sensitive manifests, and measured provenance used to produce the synthetic dataset.
- [ ] Keep the Nebius training, evaluation, Serverless AI Job, container, resource configuration, and submission code required by the challenge.
- [ ] Do not remove provider evidence merely because it is generated; redact sensitive fields and retain the useful proof.
- [ ] Clearly distinguish the legacy Nebius L40S virtual-machine experiment from the required Nebius Serverless AI Job run.
- [ ] Remove unused infrastructure paths that did not contribute to the final project, including obsolete cloud deployment code and self-hosted local-model deployment instructions.

## Images and execution evidence

- [ ] Retain useful Nebius screenshots and challenge evidence.
- [ ] Redact resource names, instance identifiers, account information, sensitive operational timestamps, and any unrelated browser or desktop content. Non-sensitive relative or execution timestamps may remain as evidence.
- [ ] Preserve non-personal content-provenance metadata only after confirming that it contains no user or account identifiers.
- [ ] Verify each redacted image visually at full resolution before staging it.
- [ ] Captions state exactly what the image proves and do not present a virtual-machine run as a Serverless AI Job run.
- [ ] Store large screenshots and media with Git LFS; do not commit temporary renders or discarded variants.
- [ ] Add redacted proof of the completed Serverless AI Job: final status, relevant logs, timestamps, resource configuration, runtime, and cost.

## Dataset publication

- [ ] Publish only validated synthetic or otherwise public data; never publish private conversations or personal data.
- [ ] Exclude raw endpoint responses, rejected rows, credentials, and the fixed evaluation holdout.
- [ ] Scan every row for PII, secrets, URLs, emails, phone numbers, credentials, private records, placeholders, and accidental provider metadata.
- [ ] Verify schema validity, action balance, duplicate rate, prompt diversity, timing buckets, row counts, and file hashes.
- [ ] The dataset card states provenance, intended use, limitations, row counts, splits, license, and data policy.
- [ ] The public dataset repository has one clean root commit and no inherited or placeholder history.
- [ ] Verify the public train and validation hashes against the retained local package.
- [ ] Add the approved public dataset identifier and immutable revision to reproduction instructions only after disclosure review.

## Model and generated artifacts

- [ ] Keep the approved training dataset, PEFT adapter, evaluation reports, training log, and executable GGUF locally until every external upload and hash is verified.
- [ ] Never delete a final executable or adapter merely because it can theoretically be regenerated.
- [ ] Do not commit model weights, adapters, checkpoints, private reports, or GGUF binaries to Git.
- [ ] Publish reviewed model artifacts through an appropriate public model repository with immutable revisions and recorded SHA-256 hashes.
- [ ] Reject incomplete model downloads and files ending in `.partial` as executable artifacts.
- [ ] Do not commit caches, bytecode, virtual environments, temporary renders, generated package caches, or test caches.
- [ ] Cleanup commands remove caches only and never remove retained datasets, adapters, runs, reports, or model binaries.

## README and reproducibility

- [ ] `README.md` explains the problem, architecture, data flow, training, evaluation, demo, security policy, and current challenge status.
- [ ] Setup instructions use only public, standard tools and begin from a fresh clone.
- [ ] The README includes the Docker build, hardware configuration, expected outputs, quality gates, approximate runtime, and approximate cost.
- [ ] Reproduction commands use public resources; account-scoped registry and subnet values remain neutral placeholders.
- [ ] Public dataset and model identifiers use approved exact values before final challenge submission.
- [ ] The README links to the public dataset, model artifacts, proof of execution, technical post, and video when each is ready.
- [ ] Commands have been executed in a clean environment; do not claim reproducibility based only on the original workstation.

## Nebius challenge requirements

- [ ] The GitHub repository is public and contains the Nebius Serverless AI code, Dockerfile or public image reference, README, and recognized open-source license.
- [ ] The repository contains no committed secrets or private data.
- [ ] A Nebius Serverless AI Job or Endpoint has actually run, and redacted proof is public.
- [ ] Measured job status, timestamps, image digest, runtime, outputs, and evaluation results are recorded; approximate cost is clearly labeled until settled billing is available.
- [ ] The technical post is at least 600 words, links to the repository, uses `#NebiusServerlessChallenge`, and explains the problem, architecture, implementation, and results.
- [ ] The optional 3–10 minute public walkthrough video is linked when available.

## Git history and GitHub publication

- [ ] Build the public branch from a clean orphan root commit; do not inherit private or legacy history.
- [ ] Use a meaningful lowercase commit message beginning with `feat:`, `fix:`, or `chore:`.
- [ ] Confirm `git rev-list --count HEAD` returns `1` for the initial publication.
- [ ] Confirm the complete root tree contains only intended public files.
- [ ] Push only after explicit approval.
- [ ] Verify GitHub reports the repository as public, `main` as the default branch, one visible commit, and `Jiply` as the author.
- [ ] If sensitive content was ever pushed, immediately make the repository private. A force-push alone does not guarantee dangling commits are inaccessible.
- [ ] To purge exposed objects immediately, delete and recreate the GitHub repository at the same name, then push only the clean root commit.
- [ ] Verify every displaced commit SHA returns `404` or `422` from GitHub after recreation.
- [ ] Verify the repository description is a complete, accurate sentence consistent with other public `Jiply` repositories.

## Final independent audit

- [ ] Clone the public repository into a new empty directory without relying on the original checkout.
- [ ] Confirm the fresh clone has one commit, a clean working tree, correct license, correct branding, and correctly rendered LFS assets.
- [ ] Repeat exhaustive searches for legacy branding, unrelated organizations, personal identifiers, internal tooling, local paths, credentials, and cloud resource IDs.
- [ ] Run the full automated test suite and record the result.
- [ ] Open every README link and inspect every public image, dataset card, model card, trace, and manifest.
- [ ] Confirm the public repository and external artifacts contain everything required for the challenge—and nothing private or workstation-specific.
