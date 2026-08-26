# GitHub and Zenodo release checklist

## Before creating the repository

- [ ] Confirm author names, order, affiliations, and ORCID identifiers.
- [ ] Confirm all authors approve the mixed MIT / CC BY 4.0 licensing.
- [ ] Confirm the journal permits public posting of the manuscript draft; remove `manuscript/` if necessary.
- [ ] Do not add Kaggle-derived CSVs without documented redistribution permission.
- [ ] Run `python tools/generate_manifest.py` and then `python tools/verify_release.py`.

## GitHub

- [ ] Create an empty public repository.
- [ ] Upload the *contents* of this folder so `README.md` is at repository root.
- [ ] Add the repository URL to `CITATION.cff` (`repository-code`) and `.zenodo.json` (`related_identifiers`).
- [ ] Commit and push.
- [ ] Confirm notebooks and figures render correctly on GitHub.
- [ ] Tag the release as `v1.0.0` and use title `CVD XAI benchmark reproducibility package v1.0.0`.

## Zenodo

- [ ] Connect the GitHub repository to Zenodo and enable archiving.
- [ ] Create the GitHub `v1.0.0` release.
- [ ] Check creator order, title, description, keywords, access, and licence on the Zenodo draft/record.
- [ ] Publish and copy both the version DOI and concept DOI.
- [ ] Add the version DOI to `CITATION.cff`, the GitHub About panel, manuscript, supplementary material, and data/code availability statements.
- [ ] For future changes, create a new semantic version and GitHub release rather than replacing the deposited archive.

## Suggested manuscript statement

> The executed notebooks, corrected analysis scripts, model predictions, statistical outputs, figures, and supplementary reproducibility materials are available at GitHub: [GITHUB URL] and archived in Zenodo: https://doi.org/[ZENODO DOI]. The source cardiovascular disease dataset is available separately from Kaggle under the source provider's terms.
